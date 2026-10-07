#!/usr/bin/env bash
# Experiment commands. Run individual commands from the repository root.
# Do not run this entire file unless you intend to train every experiment.
# Datasets: ./data; outputs: ./outputs. Change --seed=0 to 1 or 2.

# ========================================================================
# pointmaze-medium
# ========================================================================

# pointmaze-medium (bc)
python main.py --seed=0 --dataset_path=data/pointmaze-medium-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-bc --wandb_mode=offline \
  --env_name=pointmaze-medium --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/gcfbc.py \
  --train_steps=1000000 --eval_interval=100000 '--agent.actor_hidden_dims=(1024,1024,1024,1024)' \
  --agent.lr=0.0003 --agent.discount=0.99

# pointmaze-medium (mc)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-medium-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-mc --wandb_mode=offline \
  --env_name=pointmaze-medium --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99

# pointmaze-medium (td2)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-medium-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-td2 --wandb_mode=offline \
  --env_name=pointmaze-medium --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=3000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=TD2_CFM --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99

# pointmaze-medium (mchc)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-medium-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-mchc --wandb_mode=offline \
  --env_name=pointmaze-medium --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC_HC_TD2 --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99 --agent.final_discount=0.998

# pointmaze-medium (dyn)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-medium-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-dyn --wandb_mode=offline \
  --env_name=pointmaze-medium --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/onestep.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=ONESTEP_CFM --agent.lr=0.0001 \
  --agent.discount=0.99 --agent.history_size=4 --agent.sample_max_steps=200 \
  --augmentation_mode=onestep_ram_cache

# ========================================================================
# pointmaze-large
# ========================================================================

# pointmaze-large (bc)
python main.py --seed=0 --dataset_path=data/pointmaze-large-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-bc --wandb_mode=offline \
  --env_name=pointmaze-large --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/gcfbc.py \
  --train_steps=1000000 --eval_interval=100000 '--agent.actor_hidden_dims=(1024,1024,1024,1024)' \
  --agent.lr=0.0003 --agent.discount=0.99

# pointmaze-large (mc)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-large-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-mc --wandb_mode=offline \
  --env_name=pointmaze-large --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99

# pointmaze-large (td2)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-large-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-td2 --wandb_mode=offline \
  --env_name=pointmaze-large --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=3000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=TD2_CFM --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99

# pointmaze-large (mchc)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-large-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-mchc --wandb_mode=offline \
  --env_name=pointmaze-large --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC_HC_TD2 --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99 --agent.final_discount=0.998

# pointmaze-large (dyn)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-large-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-dyn --wandb_mode=offline \
  --env_name=pointmaze-large --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/onestep.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=ONESTEP_CFM --agent.lr=0.0001 \
  --agent.discount=0.99 --agent.history_size=4 --agent.sample_max_steps=200 \
  --augmentation_mode=onestep_ram_cache

# ========================================================================
# pointmaze-giant
# ========================================================================

# pointmaze-giant (bc)
python main.py --seed=0 --dataset_path=data/pointmaze-giant-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-bc --wandb_mode=offline \
  --env_name=pointmaze-giant --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/gcfbc.py \
  --train_steps=1000000 --eval_interval=100000 '--agent.actor_hidden_dims=(1024,1024,1024,1024)' \
  --agent.lr=0.0003 --agent.discount=0.99

# pointmaze-giant (mc)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-giant-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-mc --wandb_mode=offline \
  --env_name=pointmaze-giant --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99

# pointmaze-giant (td2)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-giant-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-td2 --wandb_mode=offline \
  --env_name=pointmaze-giant --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=3000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=TD2_CFM --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99

# pointmaze-giant (mchc)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-giant-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-mchc --wandb_mode=offline \
  --env_name=pointmaze-giant --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC_HC_TD2 --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99 --agent.final_discount=0.998

# pointmaze-giant (dyn)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-giant-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-dyn --wandb_mode=offline \
  --env_name=pointmaze-giant --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/onestep.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=ONESTEP_CFM --agent.lr=0.0001 \
  --agent.discount=0.99 --agent.history_size=4 --agent.sample_max_steps=200 \
  --augmentation_mode=onestep_ram_cache

# ========================================================================
# pointmaze-teleport
# ========================================================================

# pointmaze-teleport (bc)
python main.py --seed=0 --dataset_path=data/pointmaze-teleport-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-bc --wandb_mode=offline \
  --env_name=pointmaze-teleport --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/gcfbc.py \
  --train_steps=1000000 --eval_interval=100000 '--agent.actor_hidden_dims=(1024,1024,1024,1024)' \
  --agent.lr=0.0003 --agent.discount=0.99

# pointmaze-teleport (mc)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-teleport-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-mc --wandb_mode=offline \
  --env_name=pointmaze-teleport --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99

# pointmaze-teleport (td2)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-teleport-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-td2 --wandb_mode=offline \
  --env_name=pointmaze-teleport --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=3000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=TD2_CFM --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99

# pointmaze-teleport (mchc)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-teleport-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-mchc --wandb_mode=offline \
  --env_name=pointmaze-teleport --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC_HC_TD2 --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.99 --agent.final_discount=0.998

# pointmaze-teleport (dyn)
python main_gen.py --seed=0 --dataset_path=data/pointmaze-teleport-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=pointmaze-dyn --wandb_mode=offline \
  --env_name=pointmaze-teleport --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --log_interval=2000 --agent=agents/onestep.py \
  --policy=agents/gcfbc.py --policy_train_steps=1000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=50000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=ONESTEP_CFM --agent.lr=0.0001 \
  --agent.discount=0.99 --agent.history_size=4 --agent.sample_max_steps=200 \
  --augmentation_mode=onestep_ram_cache

# ========================================================================
# visual-antmaze-medium
# ========================================================================

# visual-antmaze-medium (bc)
python main_gen.py --seed=0 --dataset_path=data/visual-antmaze-medium-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=visual-bc --wandb_mode=offline \
  --env_name=visual-antmaze-medium --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --agent=agents/tdflow.py --policy=agents/gcfbc.py \
  --policy_train_steps=500000 --policy_eval_interval=100000 --train_steps=0 \
  --eval_interval=50000 --gen_prop=0.0 --num_eval_samples_per_state=32 \
  '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.batch_size=256 --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC --agent.lr=0.0001 \
  --agent.discount=0.99 --encoder=agents/vae.py --encoder_train_steps=100000 \
  --encoder_num_eval_states=2 --encoder.beta=0.0 --encoder.deterministic=True \
  --encoder.lr=0.0001

# visual-antmaze-medium (mc)
python main_gen.py --seed=0 --dataset_path=data/visual-antmaze-medium-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=visual-mc --wandb_mode=offline \
  --env_name=visual-antmaze-medium --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --agent=agents/tdflow.py --policy=agents/gcfbc.py \
  --policy_train_steps=500000 --policy_eval_interval=100000 --train_steps=1000000 \
  --eval_interval=50000 --gen_prop=0.5 --num_eval_samples_per_state=32 \
  '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.batch_size=256 --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC --agent.lr=0.0001 \
  --agent.discount=0.99 --encoder=agents/vae.py --encoder_train_steps=100000 \
  --encoder_num_eval_states=2 --encoder.beta=0.0 --encoder.deterministic=True \
  --encoder.lr=0.0001

# visual-antmaze-medium (td2)
python main_gen.py --seed=0 --dataset_path=data/visual-antmaze-medium-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=visual-td2 --wandb_mode=offline \
  --env_name=visual-antmaze-medium --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --agent=agents/tdflow.py --policy=agents/gcfbc.py \
  --policy_train_steps=500000 --policy_eval_interval=100000 --train_steps=3000000 \
  --eval_interval=50000 --gen_prop=0.5 --num_eval_samples_per_state=32 \
  '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.batch_size=256 --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=TD2_CFM --agent.lr=0.0001 \
  --agent.discount=0.99 --encoder=agents/vae.py --encoder_train_steps=100000 \
  --encoder_num_eval_states=2 --encoder.beta=0.0 --encoder.deterministic=True \
  --encoder.lr=0.0001

# visual-antmaze-medium (mchc)
python main_gen.py --seed=0 --dataset_path=data/visual-antmaze-medium-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=visual-mchc --wandb_mode=offline \
  --env_name=visual-antmaze-medium --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --agent=agents/tdflow.py --policy=agents/gcfbc.py \
  --policy_train_steps=500000 --policy_eval_interval=100000 --train_steps=1000000 \
  --eval_interval=50000 --gen_prop=0.5 --num_eval_samples_per_state=32 \
  '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.batch_size=256 --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC_HC_TD2 --agent.lr=0.0001 \
  --agent.discount=0.99 --agent.final_discount=0.995 \
  --encoder=agents/vae.py --encoder_train_steps=100000 --encoder_num_eval_states=2 \
  --encoder.beta=0.0 --encoder.deterministic=True --encoder.lr=0.0001

# ========================================================================
# visual-antmaze-large
# ========================================================================

# visual-antmaze-large (bc)
python main_gen.py --seed=0 --dataset_path=data/visual-antmaze-large-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=visual-bc --wandb_mode=offline \
  --env_name=visual-antmaze-large --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --agent=agents/tdflow.py --policy=agents/gcfbc.py \
  --policy_train_steps=500000 --policy_eval_interval=100000 --train_steps=0 \
  --eval_interval=50000 --gen_prop=0.0 --num_eval_samples_per_state=32 \
  '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.batch_size=256 --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC --agent.lr=0.0001 \
  --agent.discount=0.99 --encoder=agents/vae.py --encoder_train_steps=100000 \
  --encoder_num_eval_states=2 --encoder.beta=0.0 --encoder.deterministic=True \
  --encoder.lr=0.0001

# visual-antmaze-large (mc)
python main_gen.py --seed=0 --dataset_path=data/visual-antmaze-large-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=visual-mc --wandb_mode=offline \
  --env_name=visual-antmaze-large --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --agent=agents/tdflow.py --policy=agents/gcfbc.py \
  --policy_train_steps=500000 --policy_eval_interval=100000 --train_steps=1000000 \
  --eval_interval=50000 --gen_prop=0.5 --num_eval_samples_per_state=32 \
  '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.batch_size=256 --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC --agent.lr=0.0001 \
  --agent.discount=0.99 --encoder=agents/vae.py --encoder_train_steps=100000 \
  --encoder_num_eval_states=2 --encoder.beta=0.0 --encoder.deterministic=True \
  --encoder.lr=0.0001

# visual-antmaze-large (td2)
python main_gen.py --seed=0 --dataset_path=data/visual-antmaze-large-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=visual-td2 --wandb_mode=offline \
  --env_name=visual-antmaze-large --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --agent=agents/tdflow.py --policy=agents/gcfbc.py \
  --policy_train_steps=500000 --policy_eval_interval=100000 --train_steps=3000000 \
  --eval_interval=50000 --gen_prop=0.5 --num_eval_samples_per_state=32 \
  '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.batch_size=256 --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=TD2_CFM --agent.lr=0.0001 \
  --agent.discount=0.99 --encoder=agents/vae.py --encoder_train_steps=100000 \
  --encoder_num_eval_states=2 --encoder.beta=0.0 --encoder.deterministic=True \
  --encoder.lr=0.0001

# visual-antmaze-large (mchc)
python main_gen.py --seed=0 --dataset_path=data/visual-antmaze-large-stitch-terminal-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=visual-mchc --wandb_mode=offline \
  --env_name=visual-antmaze-large --eval_episodes=50 --video_episodes=0 \
  --save_interval=500000 --agent=agents/tdflow.py --policy=agents/gcfbc.py \
  --policy_train_steps=500000 --policy_eval_interval=100000 --train_steps=1000000 \
  --eval_interval=50000 --gen_prop=0.5 --num_eval_samples_per_state=32 \
  '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.batch_size=256 --policy.lr=0.0003 \
  --policy.discount=0.99 --agent.loss_type=MC_HC_TD2 --agent.lr=0.0001 \
  --agent.discount=0.99 --agent.final_discount=0.995 \
  --encoder=agents/vae.py --encoder_train_steps=100000 --encoder_num_eval_states=2 \
  --encoder.beta=0.0 --encoder.deterministic=True --encoder.lr=0.0001

# ========================================================================
# cube-double
# ========================================================================

# cube-double (bc-play)
python main.py --seed=0 --dataset_path=data/cube-double-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-bc-play --wandb_mode=offline \
  --env_name=cube-double-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=0 --agent=agents/gcfbc.py \
  --train_steps=2000000 --eval_interval=100000 '--agent.actor_hidden_dims=(1024,1024,1024,1024)' \
  --agent.lr=3e-05 --agent.discount=0.99 --agent.actor_geom_sample=True \
  --agent.action_chunk_len=10 --goal_representation=oracle

# cube-double (bc-stitch)
python main.py --seed=0 --dataset_path=data/cube-double-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-bc-stitch --wandb_mode=offline \
  --env_name=cube-double-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=1 --agent=agents/gcfbc.py \
  --train_steps=2000000 --eval_interval=100000 '--agent.actor_hidden_dims=(1024,1024,1024,1024)' \
  --agent.lr=3e-05 --agent.discount=0.99 --agent.actor_geom_sample=True \
  --agent.action_chunk_len=10 --goal_representation=oracle

# cube-double (td2)
python main_gen.py --seed=0 --dataset_path=data/cube-double-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-td2 --wandb_mode=offline \
  --env_name=cube-double-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=1 --agent=agents/chunk_tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=2000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=100000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=3e-05 \
  --policy.discount=0.99 --policy.actor_geom_sample=True --policy.action_chunk_len=10 \
  --policy.goal_representation=oracle --agent.loss_type=TD2_CFM --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.9 \
  --agent.action_chunk_size=10 --agent.bootstrap_stride=10 \
  --agent.bootstrap_continuation_mode=policy_prior --agent.final_discount=0.99 --action_prior=agents/gcfbc.py \
  --action_prior_train_steps=1000000 --metric_discount=0.99 '--action_prior.actor_hidden_dims=(1024,1024,1024,1024)' \
  --action_prior.lr=3e-05 --action_prior.discount=0.99 --action_prior.actor_geom_sample=True \
  --action_prior.action_chunk_len=10 --action_prior.goal_representation=oracle --action_prior.single_input_to_actor=True

# cube-double (mchc)
python main_gen.py --seed=0 --dataset_path=data/cube-double-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-mchc --wandb_mode=offline \
  --env_name=cube-double-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=1 --agent=agents/chunk_tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=2000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=100000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=3e-05 \
  --policy.discount=0.99 --policy.actor_geom_sample=True --policy.action_chunk_len=10 \
  --policy.goal_representation=oracle --agent.loss_type=MC_HC_TD2 --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.98 --agent.final_discount=0.99 \
  --agent.action_chunk_size=10 --agent.bootstrap_stride=10 --agent.bootstrap_continuation_mode=policy_prior \
  --agent.mchc_sampling_unit=chunk --action_prior=agents/gcfbc.py --action_prior_train_steps=1000000 \
  --metric_discount=0.99 '--action_prior.actor_hidden_dims=(1024,1024,1024,1024)' --action_prior.lr=3e-05 \
  --action_prior.discount=0.99 --action_prior.actor_geom_sample=True --action_prior.action_chunk_len=10 \
  --action_prior.goal_representation=oracle --action_prior.single_input_to_actor=True

# ========================================================================
# cube-triple
# ========================================================================

# cube-triple (bc-play)
python main.py --seed=0 --dataset_path=data/cube-triple-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-bc-play --wandb_mode=offline \
  --env_name=cube-triple-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=0 --agent=agents/gcfbc.py \
  --train_steps=2000000 --eval_interval=100000 '--agent.actor_hidden_dims=(1024,1024,1024,1024)' \
  --agent.lr=3e-05 --agent.discount=0.99 --agent.actor_geom_sample=True \
  --agent.action_chunk_len=10 --goal_representation=oracle

# cube-triple (bc-stitch)
python main.py --seed=0 --dataset_path=data/cube-triple-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-bc-stitch --wandb_mode=offline \
  --env_name=cube-triple-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=1 --agent=agents/gcfbc.py \
  --train_steps=2000000 --eval_interval=100000 '--agent.actor_hidden_dims=(1024,1024,1024,1024)' \
  --agent.lr=3e-05 --agent.discount=0.99 --agent.actor_geom_sample=True \
  --agent.action_chunk_len=10 --goal_representation=oracle

# cube-triple (td2)
python main_gen.py --seed=0 --dataset_path=data/cube-triple-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-td2 --wandb_mode=offline \
  --env_name=cube-triple-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=1 --agent=agents/chunk_tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=2000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=100000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=3e-05 \
  --policy.discount=0.99 --policy.actor_geom_sample=True --policy.action_chunk_len=10 \
  --policy.goal_representation=oracle --agent.loss_type=TD2_CFM --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.9 \
  --agent.action_chunk_size=10 --agent.bootstrap_stride=10 \
  --agent.bootstrap_continuation_mode=policy_prior --agent.final_discount=0.99 --action_prior=agents/gcfbc.py \
  --action_prior_train_steps=1000000 --metric_discount=0.99 '--action_prior.actor_hidden_dims=(1024,1024,1024,1024)' \
  --action_prior.lr=3e-05 --action_prior.discount=0.99 --action_prior.actor_geom_sample=True \
  --action_prior.action_chunk_len=10 --action_prior.goal_representation=oracle --action_prior.single_input_to_actor=True

# cube-triple (mchc)
python main_gen.py --seed=0 --dataset_path=data/cube-triple-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-mchc --wandb_mode=offline \
  --env_name=cube-triple-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=1 --agent=agents/chunk_tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=2000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=100000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=3e-05 \
  --policy.discount=0.99 --policy.actor_geom_sample=True --policy.action_chunk_len=10 \
  --policy.goal_representation=oracle --agent.loss_type=MC_HC_TD2 --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.98 --agent.final_discount=0.99 \
  --agent.action_chunk_size=10 --agent.bootstrap_stride=10 --agent.bootstrap_continuation_mode=policy_prior \
  --agent.mchc_sampling_unit=chunk --action_prior=agents/gcfbc.py --action_prior_train_steps=1000000 \
  --metric_discount=0.99 '--action_prior.actor_hidden_dims=(1024,1024,1024,1024)' --action_prior.lr=3e-05 \
  --action_prior.discount=0.99 --action_prior.actor_geom_sample=True --action_prior.action_chunk_len=10 \
  --action_prior.goal_representation=oracle --action_prior.single_input_to_actor=True

# ========================================================================
# puzzle-4x4
# ========================================================================

# puzzle-4x4 (bc-play)
python main.py --seed=0 --dataset_path=data/puzzle-4x4-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-bc-play --wandb_mode=offline \
  --env_name=puzzle-4x4-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=0 --agent=agents/gcfbc.py \
  --train_steps=2000000 --eval_interval=100000 '--agent.actor_hidden_dims=(1024,1024,1024,1024)' \
  --agent.lr=3e-05 --agent.discount=0.99 --agent.actor_geom_sample=True \
  --agent.action_chunk_len=10 --goal_representation=oracle

# puzzle-4x4 (bc-stitch)
python main.py --seed=0 --dataset_path=data/puzzle-4x4-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-bc-stitch --wandb_mode=offline \
  --env_name=puzzle-4x4-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=1 --agent=agents/gcfbc.py \
  --train_steps=2000000 --eval_interval=100000 '--agent.actor_hidden_dims=(1024,1024,1024,1024)' \
  --agent.lr=3e-05 --agent.discount=0.99 --agent.actor_geom_sample=True \
  --agent.action_chunk_len=10 --goal_representation=oracle

# puzzle-4x4 (td2)
python main_gen.py --seed=0 --dataset_path=data/puzzle-4x4-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-td2 --wandb_mode=offline \
  --env_name=puzzle-4x4-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=1 --agent=agents/chunk_tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=2000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=100000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=3e-05 \
  --policy.discount=0.99 --policy.actor_geom_sample=True --policy.action_chunk_len=10 \
  --policy.goal_representation=oracle --agent.loss_type=TD2_CFM --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.9 \
  --agent.action_chunk_size=10 --agent.bootstrap_stride=10 \
  --agent.bootstrap_continuation_mode=policy_prior --agent.final_discount=0.99 --action_prior=agents/gcfbc.py \
  --action_prior_train_steps=1000000 --metric_discount=0.99 '--action_prior.actor_hidden_dims=(1024,1024,1024,1024)' \
  --action_prior.lr=3e-05 --action_prior.discount=0.99 --action_prior.actor_geom_sample=True \
  --action_prior.action_chunk_len=10 --action_prior.goal_representation=oracle --action_prior.single_input_to_actor=True

# puzzle-4x4 (mchc)
python main_gen.py --seed=0 --dataset_path=data/puzzle-4x4-play-subtasks-v0.npz --save_dir=outputs \
  --project=td-aug --run_group=manipulation-mchc --wandb_mode=offline \
  --env_name=puzzle-4x4-v0 --eval_episodes=50 --video_episodes=0 \
  --save_interval=100000 --dataset_subtask_split_every=1 --agent=agents/chunk_tdflow.py \
  --policy=agents/gcfbc.py --policy_train_steps=2000000 --policy_eval_interval=100000 \
  --train_steps=1000000 --eval_interval=100000 --gen_prop=0.5 \
  --num_eval_samples_per_state=64 '--policy.actor_hidden_dims=(1024,1024,1024,1024)' --policy.lr=3e-05 \
  --policy.discount=0.99 --policy.actor_geom_sample=True --policy.action_chunk_len=10 \
  --policy.goal_representation=oracle --agent.loss_type=MC_HC_TD2 --agent.lr=0.0001 \
  --agent.normalize_observations=True --agent.discount=0.98 --agent.final_discount=0.99 \
  --agent.action_chunk_size=10 --agent.bootstrap_stride=10 --agent.bootstrap_continuation_mode=policy_prior \
  --agent.mchc_sampling_unit=chunk --action_prior=agents/gcfbc.py --action_prior_train_steps=1000000 \
  --metric_discount=0.99 '--action_prior.actor_hidden_dims=(1024,1024,1024,1024)' --action_prior.lr=3e-05 \
  --action_prior.discount=0.99 --action_prior.actor_geom_sample=True --action_prior.action_chunk_len=10 \
  --action_prior.goal_representation=oracle --action_prior.single_input_to_actor=True
