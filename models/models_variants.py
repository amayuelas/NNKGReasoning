#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ast import And, Not

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
import os

from models.mlp import ProjectionMLP, OrMLP, AndMLP, NotMLP
from models.attention import AndAttention

from manifolds.poincare import PoincareBall

class Variants(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo, test_batch_size=1,
                 use_cuda=False,
                 query_name_dict=None,
                 mlp_mode=None):
        super(Variants, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size,
                                                                               1).cuda() if self.use_cuda else torch.arange(
            nentity).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
        self.query_name_dict = query_name_dict
        self.layers = mlp_mode
        # ----------------------------------------
        # hyper embedding space
        self.poincareball = PoincareBall()
        self.curvature = 1
        # ----------------------------------------

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))  # center for entities

        if self.geo == 'mlp2vector':
            self.notNN1 = NotMLP(self.layers, self.entity_dim)
            self.projectionNN1 = ProjectionMLP(self.layers, self.entity_dim)
            self.andNN1 = AndMLP(self.layers, self.entity_dim)
            self.orNN1 = OrMLP(self.layers, self.entity_dim)

            self.notNN2 = NotMLP(self.layers, self.entity_dim)
            self.projectionNN2 = ProjectionMLP(self.layers, self.entity_dim)
            self.andNN2 = AndMLP(self.layers, self.entity_dim)
            self.orNN2 = OrMLP(self.layers, self.entity_dim)
        
        elif self.geo == 'mlpAttention':
            self.notNN = NotMLP(self.layers, self.entity_dim)
            self.projectionNN = ProjectionMLP(self.layers, self.entity_dim)
            self.andNN = AndAttention(self.layers, self.entity_dim, 1)
            self.orNN = OrMLP(self.layers, self.entity_dim)

        elif self.geo == 'mlpHyperE':
            self.notNN = NotMLP(self.layers, self.entity_dim)
            self.projectionNN = ProjectionMLP(self.layers, self.entity_dim)
            self.andNN = AndMLP(self.layers, self.entity_dim)
            self.orNN = OrMLP(self.layers, self.entity_dim)


        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        
        if self.geo == 'mlpHyperE':
            return self.forward_hyperE(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                    batch_idxs_dict)
        else:
            return self.forward_mlp(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                    batch_idxs_dict)

    def embed_query_mlp2vector(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query_mlp2vector(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    # Negation
                    assert (queries[:, idx] == -2).all()
                    embedding_1 = self.notNN1(embedding)
                    embedding_2 = self.notNN2(embedding)
                    embedding = (embedding_1 + embedding_2) / 2
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding_1 = self.projectionNN1(embedding, r_embedding)
                    embedding_2 = self.projectionNN2(embedding, r_embedding)
                    embedding = (embedding_1 + embedding_2) / 2
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_mlp2vector(queries, query_structure[i], idx)
                embedding_list.append(embedding)

            vector = embedding_list[0]
            for i in range(1, len(embedding_list)):
                vector_1 = self.andNN1(vector, embedding_list[i])
                vector_2 = self.andNN2(vector, embedding_list[i])
                vector = (vector_1 + vector_2) / 2
            embedding = vector

        return embedding, idx

    def embed_query(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    # Negation
                    assert (queries[:, idx] == -2).all()
                    embedding = self.notNN(embedding)
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projectionNN(embedding, r_embedding)
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query(queries, query_structure[i], idx)
                embedding_list.append(embedding)

            vector = embedding_list[0]
            for i in range(1, len(embedding_list)):
                vector = self.andNN(vector, embedding_list[i])
            embedding = vector

        return embedding, idx


    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_logit_mlp(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_mlp(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                if self.geo == 'mlp2vector':
                    center_embedding, _ = self.embed_query_mlp2vector(
                        self.transform_union_query(batch_queries_dict[query_structure],
                                                query_structure),
                        self.transform_union_structure(query_structure), 0)
                elif self.geo == 'mlpAttention':
                    center_embedding, _ = self.embed_query(
                        self.transform_union_query(batch_queries_dict[query_structure],
                                                query_structure),
                        self.transform_union_structure(query_structure), 0)

                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                if self.geo == 'mlp2vector':
                    center_embedding, _ = self.embed_query_mlp2vector(batch_queries_dict[query_structure], query_structure, 0)
                elif self.geo == 'mlpAttention':
                    center_embedding, _ = self.embed_query(batch_queries_dict[query_structure], query_structure, 0)
                
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0] // 2, 2,
                                                                           1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_mlp(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_mlp(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1)).view(batch_size,
                                                                                                     negative_size, -1)
                negative_logit = self.cal_logit_mlp(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_union.view(-1)).view(batch_size, 1,
                                                                                                   negative_size, -1)
                negative_union_logit = self.cal_logit_mlp(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs


    def cal_logit_hyperE(self, entity_embedding, query_embedding):
        entity_embedding = self.poincareball.proj(self.poincareball.expmap0(entity_embedding, self.curvature),
                                                  self.curvature)
        query_embedding = self.poincareball.proj(self.poincareball.expmap0(query_embedding, self.curvature),
                                                 self.curvature)
        distance = self.poincareball.sqdist(entity_embedding, query_embedding, self.curvature)
        logit = self.gamma - distance  # torch.norm(distance, p=1, dim=-1)
        logit = torch.squeeze(logit, -1)
        return logit

    def forward_hyperE(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query(
                    self.transform_union_query(batch_queries_dict[query_structure],
                                               query_structure),
                    self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query(batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0] // 2, 2,
                                                                           1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_hyperE(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_hyperE(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1)).view(batch_size,
                                                                                                     negative_size, -1)
                negative_logit = self.cal_logit_hyperE(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_union.view(-1)).view(batch_size, 1,
                                                                                                   negative_size, -1)
                negative_union_logit = self.cal_logit_hyperE(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries):  # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample,
                                                                      subsampling_weight, batch_queries_dict,
                                                                      batch_idxs_dict)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False,
                  save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader,
                                                                                      disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(
                        argsort) == args.test_batch_size:  # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort,
                                               model.batch_entity_range)  # achieve the ranking of all entities
                else:  # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1).cuda()
                                                   )  # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1)
                                                   )  # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(
                        zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics