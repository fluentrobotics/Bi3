# Bi3 Setup

Bi3 JSON files are expected under:

```
/home/socnav/Desktop/Bi3/jsons/{um,laas}/{experiment}/{condition}.json
```

Create the trajectory prediction files with:

```
./.venv/bin/python datasets/bi3/create_data_npys.py --bi3-root /home/socnav/Desktop/Bi3 --output-root /home/socnav/Desktop/Bi3/trajectory_prediction --overwrite
```

The converter writes `train`, `val`, and `test` subdirectories under
`/home/socnav/Desktop/Bi3/trajectory_prediction`. Each source JSON produces a
TrajNet++-style `.ndjson`, a model-ready `.npy`, and a matching
`*_agent_types.npy` file. The dataset uses 9 observed timesteps and 12 future
timesteps at 2.5 Hz.

Training AutoBot-Ego on Bi3:

```
./.venv/bin/python train.py --exp-id bi3_ego --seed 1 --dataset bi3 --model-type Autobot-Ego --num-modes 6 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path /home/socnav/Desktop/Bi3/trajectory_prediction
```

Training AutoBot-Joint on Bi3:

```
./.venv/bin/python train.py --exp-id bi3_joint --seed 1 --dataset bi3 --model-type Autobot-Joint --num-modes 6 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path /home/socnav/Desktop/Bi3/trajectory_prediction
```

The same default Bi3 Joint setup is available as a YAML config:

```
./.venv/bin/python train.py --config default_train.yaml
```

Evaluate on the validation split by default, or on test with `--eval-split test`:

```
./.venv/bin/python evaluate.py --dataset-path /home/socnav/Desktop/Bi3/trajectory_prediction --models-path results/bi3/{exp_name}/best_models_ade.pth --batch-size 64 --eval-split test
```

Or use the default evaluation config after training with `default_train.yaml`:

```
./.venv/bin/python evaluate.py --config default_eval.yaml
```

Visualize one prediction from a trained model:

```
./.venv/bin/python useful_scripts/visualize_prediction.py --models-path results/bi3/{exp_name}/best_models_ade.pth --dataset-path /home/socnav/Desktop/Bi3/trajectory_prediction --split test --index 0 --output bi3_prediction.png
```

If `--index` is omitted, the script samples a random datapoint.
