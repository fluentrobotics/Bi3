# Bi3 Official Repository

This repository contains training and evaluation code for the [Bi3](https://fluentrobotics.com/bi3dataset/) dataset. It is a modified version of the [AutoBots](https://arxiv.org/abs/2104.00563) repository which supports Bi3.

### Getting Started

1. Create a python 3.7 environment. I use Miniconda3 and create with 
`conda create --name AutoBots python=3.7`
2. Run `pip install -r requirements.txt`

### Setting up and training on Bi3
Convert Bi3 JSON files to trajectory prediction files:
```
./{path to venv}/bin/python datasets/bi3/create_data_npys.py --bi3-root {path to Bi3 root} --output-root {path to desired output root}/Bi3/trajectory_prediction --overwrite
```

Training AutoBot-Ego on Bi3:
```
./{path to venv}/bin/python train.py --exp-id bi3_ego --seed 1 --dataset bi3 --model-type Autobot-Ego --num-modes 6 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path {path to output root}/Bi3/trajectory_prediction
```

Training AutoBot-Joint on Bi3:
```
./{path to venv}/bin/python train.py --exp-id bi3_joint --seed 1 --dataset bi3 --model-type Autobot-Joint --num-modes 6 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path {path to output root}/Bi3/trajectory_prediction
```

The same Bi3 Joint defaults can be run from the checked-in config:
```
./{path to venv}/bin/python train.py --config default_train.yaml
```

### Evaluating an AutoBot model

For the default Bi3 Joint config, evaluate the best-ADE checkpoint with:
```
./{path to venv}/bin/python evaluate.py --config default_eval.yaml
```



## Reference

If you use this repository, please cite our work:
```
@inproceedings{stratton2026bi3dataset,
  author    = {Stratton, Andrew and Singamaneni, Phani Teja and Goyal, Pranav and Alami, Rachid and Mavrogiannis, Christoforos},
  title     = {Bi3: A Biplatform, Bicultural, Biperson Dataset for Social Robot Navigation},
  booktitle   = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026},
}
```
