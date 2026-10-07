# Datasets

## Download

Download the pre-collected datasets from
[Hugging Face](https://huggingface.co/datasets/daniellawson9999/tdaug-datasets)
To download you can install the CLI with `pip install huggingface_hub` and run:

```bash
hf download daniellawson9999/tdaug-datasets \
  pointmaze-large-stitch-terminal-v0.npz \
  pointmaze-large-stitch-terminal-v0-val.npz \
  --repo-type dataset --local-dir data
```
Or to download everything:
```bash
hf download daniellawson9999/tdaug-datasets  --repo-type dataset --local-dir data
```
To regenerate datasets, use the commands below. 

### Files

Commands in `hyperparameters.sh` use the following names:

| Environment | Filename (add `.npz` or `-val.npz`) |
| --- | --- |
| PointMaze medium | `pointmaze-medium-stitch-terminal-v0` |
| PointMaze large | `pointmaze-large-stitch-terminal-v0` |
| PointMaze giant | `pointmaze-giant-stitch-terminal-v0` |
| PointMaze teleport | `pointmaze-teleport-stitch-terminal-v0` |
| Visual AntMaze medium | `visual-antmaze-medium-stitch-terminal-v0` |
| Visual AntMaze large | `visual-antmaze-large-stitch-terminal-v0` |
| Cube double | `cube-double-play-subtasks-v0` |
| Cube triple | `cube-triple-play-subtasks-v0` |
| Puzzle 4x4 | `puzzle-4x4-play-subtasks-v0` |

 All datasets (18 NPZ files) are approximately 6.1 GB compressed. 

Manipulation files contain subtask metadata used to split trajectories during
training. Normal play training is supported with
`dataset_subtask_split_every=0`.

## Collecting

Scripts collect maze trajectories and subtask-annotated manipulation play
data. Run them from the repository root with
`PYTHONPATH=.`. Use `MUJOCO_GL=egl` for headless rendering.

### PointMaze


Collect terminal-stitch trajectories with the scripted controller:

```bash
PYTHONPATH=. MUJOCO_GL=egl python data_gen_scripts/generate_locomaze_timesteps.py \
  --env_name=pointmaze-large-v0 --dataset_type=stitch \
  --max_episode_steps=201 --noise=0.5 --num_timesteps=1000000 \
  --save_path=/path/to/data/pointmaze-large-stitch-terminal-v0.npz
```

The collector finishes complete trajectories, so the transition count can slightly exceed
the requested timesteps. It collects validation trajectories separately, for
an additional 10% of the training step budget.


### Visual AntMaze

Use `generate_locomaze_timesteps.py` with a visual AntMaze environment and a
compatible SAC locomotion expert supplied through `--restore_path` and
`--restore_epoch`.

### Manipulation

Use `generate_manipspace.py` with `--dataset_type=play` and
`--save_subtask_metadata=True`. For example:

```bash
PYTHONPATH=. MUJOCO_GL=egl python data_gen_scripts/generate_manipspace.py \
  --env_name=cube-double-v0 --dataset_type=play \
  --save_subtask_metadata=True \
  --save_path=/path/to/data/cube-double-play-subtasks-v0.npz
```

### Other

- `generate_locomaze.py`: episode-based locomotion collection.
- `generate_locomaze_eval.py`: task-conditioned empirical maze trajectories.
- `generate_manipspace_eval.py`: cube evaluation trajectories.
- `viz_utils.py`: locomotion collection diagnostics.
