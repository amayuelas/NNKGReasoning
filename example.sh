# ------------------------------------------------------------------ # 
# Option: --geo
# --geo allows you to select the model you want to use. 
# You can select from the following models: 
# 'vec', 'box', 'beta',                           # Baselines
# 'mlp', 'mlpMixer',                              # Models
# 'mlp2vector', 'mlpAttention', 'mlpHyperE',      # Model Variants
# 'cnn', 'nln'                                    # Extra Models
# ------------------------------------------------------------------ # 


# FB15k-237
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo mlp --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up"


# FB15k
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/FB15k-betae -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo mlp --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up"


# NELL
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/NELL-betae -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo mlp --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up"


## Evaluation

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test \
  --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo mlp --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up" --checkpoint_path $CKPT_PATH
