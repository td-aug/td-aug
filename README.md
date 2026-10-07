# TD-Aug

Code for **Imagining Goals with Geometric Horizon Models for Policy Generalization**.

## Environment

You can use Conda or [uv](https://docs.astral.sh/uv/getting-started/installation/):

**Conda**

```bash
conda env create -f environment.yaml
conda activate tdaug
```

**uv**

```bash
uv venv --python 3.10 --managed-python .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

For headless rendering:
```bash
export MUJOCO_GL=egl
```

## Datasets

Download data from [Hugging Face](https://huggingface.co/datasets/daniellawson9999/tdaug-datasets)
into a folder, e.g. `data/`. See the [download instructions](data_gen_scripts/README.md#download).

Your datasets should like:

```text
data/
  pointmaze-large-stitch-terminal-v0.npz
  pointmaze-large-stitch-terminal-v0-val.npz
  visual-antmaze-medium-stitch-terminal-v0.npz
  visual-antmaze-medium-stitch-terminal-v0-val.npz
  cube-double-play-subtasks-v0.npz
  cube-double-play-subtasks-v0-val.npz
```

## Training

[hyperparameters.sh](hyperparameters.sh) lists GCFBC, MC, TD2, MCHC, and Dyn-Aug
training commands. Each command calls `main.py` or `main_gen.py`, where `main.py` is used for training policies, and `main_gen.py` trains a tdflow, and then a policy with augmented goals. Visual experiments also train an image encoder and manipulation experiments also train an action prior.

For example, to train GCFBC policy on pointmaze-large:

```bash
python main.py \
  --env_name=pointmaze-large \
  --dataset_path=data/pointmaze-large-stitch-terminal-v0.npz \
  --agent=agents/gcfbc.py \
  --agent.actor_hidden_dims='(1024,1024,1024,1024)' \
  --agent.lr=3e-4 --agent.discount=0.99 \
  --train_steps=1000000 --eval_interval=100000 --eval_episodes=50 \
  --seed=0 --save_dir=outputs --wandb_mode=offline
```
or MCHC on pointmaze-large:
```bash
python main_gen.py --seed=0 --dataset_path=data/pointmaze-large-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-mchc --wandb_mode=offline \
  --env_name=pointmaze-large --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC_HC_TD2 --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99 --agent.final_discount=0.998
```

We used `--seed=0`, `1` and `2`. Outputs are saved under
`outputs/` and use offline W&B logging. Set `--wandb_mode=online` and
`--project=YOUR_PROJECT` for online logging.

## Credits

Built on [OGBench](https://github.com/seohongpark/ogbench) and
[TRA-OGBench](https://github.com/vivekmyers/tra-ogbench).
