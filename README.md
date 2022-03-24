# Neural Methods for Logical Reasoning over Knowledge Graphs
This repo contains the code implementing the models described in [Neural Methods for Logical Reasoning over Knowledge Graphs](https://openreview.net/forum?id=tgcAoUVHRIB)

In this paper, we focus on answering multi-hop logical queries on Knowledge Graphs (KGs). To this end, we have implemented the following models. We also include the original baseline models

**Baselines**
- [x] [BetaE](https://arxiv.org/abs/2010.11465)
- [x] [Query2box](https://arxiv.org/abs/2002.05969)
- [x] [GQE](https://arxiv.org/abs/1806.01445)

**Models**
- [x] MLP: Multi-Layer Perceptron
- [x] MLPMixer: Adpated from [MLPMixer](https://arxiv.org/abs/2105.01601)

**Variants**
- [x] MLP + [Heterogeneous Hyper-Graph Embeddings](https://arxiv.org/abs/2010.10728)
- [x] MLP + [Attention Mechanism](https://arxiv.org/pdf/1706.03762.pdf)
- [x] MLP + 2 Vector Average


**How to use it**

You can find some examples on how to execute the code can be found on `examples.sh`

**Data**

To evalute the models, we have used standard evaluation datasets (FB15k, FB15k-237, NELL995) as in the BetaE paper. It can be downloaded [here](https://drive.google.com/drive/folders/1vCPaHL0RqksyVcaE_jFzpWIAe7DdeSzo?usp=sharing).

**Citations**

If you use this repo, please cite the following paper.

```
@inproceedings{
    amayuelas2022neural,
    title={Neural Methods for Logical Reasoning over Knowledge Graphs},
    author={Amayuelas, Alfonso and Zhang, Shuai and Rao, Xi Susie and Zhang, Ce},
    booktitle={International Conference on Learning Representations},
    year={2022}
}
```

**Acknowledgements**

This code is built on top of previous work from SNAP-Stanford. Check out their repo [here](https://github.com/snap-stanford/KGReasoning)
