import copy
import os
import platform

if "mac" in platform.platform():
    # macOS doesn't support EGL.
    pass
else:
    os.environ["MUJOCO_GL"] = "egl"
    if "SLURM_STEP_GPUS" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]
        os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]

import datetime
import json
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import wandb
from agents import agents
from ogbench.env_loader import make_env_and_dataset
from utils.datasets import (
    Dataset,
    GCDataset,
    HGCDataset,
    split_trajectories_by_subtasks,
)
from utils.evaluation import (
    evaluate,
)
from utils.flax_utils import restore_agent, save_agent
from utils.gen_evaluation import (
    gen_augment,
    plot_samples_xy,
)
from utils.goal_representations import (
    add_cube_oracle_representations,
    add_locomaze_oracle_representations,
    add_puzzle_oracle_representations,
    add_scene_oracle_representations,
)
from utils.log_utils import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    get_wandb_video,
    setup_wandb,
)
from utils.run_restore import load_run_flags, resolve_run_directory

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "antmaze-large-v0", "Environment name.")
flags.DEFINE_string("dataset_path", None, "Dataset path.")
flags.DEFINE_enum(
    "goal_representation",
    "full",
    ["full", "oracle"],
    "Goal representation used by the offline dataset and evaluation.",
)
flags.DEFINE_integer(
    "dataset_subtask_split_every",
    0,
    "Split loaded trajectories after every N subtask switches; 0 disables splitting.",
    lower_bound=0,
)
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("wandb_mode", "online", "wandb mode")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")
flags.DEFINE_bool("resume_training", False, "Resume GCFBC optimizer progress; train_steps is the total target.")

# Restore a trained generative model for policy augmentation.
flags.DEFINE_string("gen_model_path", None, "Path to the generative model checkpoint.")
flags.DEFINE_integer(
    "gen_model_epoch", None, "Epoch of the generative model checkpoint."
)
# proportion of data for gen model evaluation
flags.DEFINE_float(
    "gen_prop",
    0.0,
    "Proportion of batch that comes from generative model augmentation.",
)
flags.DEFINE_string(
    "learned_encoder_path", None, "Path to a learned encoder checkpoint directory."
)
flags.DEFINE_integer(
    "learned_encoder_epoch", None, "Checkpoint epoch for learned_encoder_path."
)
flags.DEFINE_integer(
    "precomputed_features", 0, "Whether the dataset contains precomputed features."
)


flags.DEFINE_integer("train_steps", 500000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_bool("eval_at_first_step", True, "Evaluate after the first update in addition to periodic evaluation.")
flags.DEFINE_integer("save_interval", 1000000, "Saving interval; 0 disables checkpoint saving.")

flags.DEFINE_integer("eval_tasks", None, "Number of tasks to evaluate (None for all).")
flags.DEFINE_list(
    "eval_task_ids",
    None,
    "Optional comma-separated one-based task IDs to evaluate.",
)
flags.DEFINE_integer("eval_episodes", 50, "Number of episodes for each task.")
flags.DEFINE_float("eval_temperature", 0, "Actor temperature for evaluation.")
flags.DEFINE_float("eval_gaussian", None, "Action Gaussian noise for evaluation.")
flags.DEFINE_integer("video_episodes", 1, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")
flags.DEFINE_integer("eval_on_cpu", 0, "Whether to evaluate on CPU.")
flags.DEFINE_integer(
    "clip_dataset_actions",
    0,
    "Whether to clip dataset actions to the open interval (-1, 1).",
)

flags.DEFINE_string("project", "comb-gen-gen", "WandB project name.")

config_flags.DEFINE_config_file("agent", "agents/gcfbc.py", lock_config=False)


def _restore_flow_model_from_run(run_dir, epoch, example_obs, example_actions):
    """Instantiate and restore a flow model from a saved run directory."""
    if epoch is None:
        raise ValueError("gen_model_epoch must be set when providing gen_model_path.")
    flag_dict = load_run_flags(run_dir)
    flow_config = copy.deepcopy(flag_dict["agent"])
    agent_name = flow_config.get("agent_name", "tdflow")
    if agent_name not in agents:
        raise ValueError(f"Unknown flow agent {agent_name} in {run_dir}")
    flow_cls = agents[agent_name]
    flow_seed = flag_dict.get("seed", 0)
    flow_model = flow_cls.create(
        flow_seed,
        example_obs,
        example_actions,
        flow_config,
    )
    flow_model = restore_agent(flow_model, run_dir, epoch)
    return flow_model, flow_config, flag_dict


def _restore_encoder_from_run(run_dir, epoch, example_obs, example_actions):
    """Instantiate and restore an encoder-like agent from a saved run directory."""
    if epoch is None:
        raise ValueError(
            "learned_encoder_epoch must be set when providing learned_encoder_path."
        )
    flag_dict = load_run_flags(run_dir)
    config = copy.deepcopy(flag_dict.get("encoder", flag_dict["agent"]))
    agent_name = config.get("agent_name")
    if agent_name not in agents:
        raise ValueError(f"Unknown encoder agent {agent_name} in {run_dir}")
    encoder_cls = agents[agent_name]
    seed = flag_dict.get("seed", 0)
    encoder = encoder_cls.create(
        seed,
        example_obs,
        example_actions,
        config,
    )
    encoder = restore_agent(encoder, run_dir, epoch)
    return encoder, config, flag_dict


def _evaluate_all_tasks(
    eval_agent,
    env,
    config,
):
    """Evaluate all requested tasks and return metrics, renders, and overall success."""
    renders = []
    eval_metrics = {}
    overall_metrics = defaultdict(list)
    task_infos = (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, "task_infos")
        else env.task_infos
    )
    if FLAGS.eval_task_ids is not None and FLAGS.eval_tasks is not None:
        raise ValueError("Set only one of --eval_task_ids or --eval_tasks.")
    if FLAGS.eval_task_ids is not None:
        task_ids = [int(task_id) for task_id in FLAGS.eval_task_ids]
        if not task_ids or any(
            task_id < 1 or task_id > len(task_infos) for task_id in task_ids
        ):
            raise ValueError(
                f"--eval_task_ids must be within [1, {len(task_infos)}]."
            )
    else:
        num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
        task_ids = list(range(1, num_tasks + 1))
    num_tasks = len(task_ids)
    num_video_episodes = FLAGS.video_episodes
    for task_id in tqdm.tqdm(task_ids):
        task_name = task_infos[task_id - 1]["task_name"]
        eval_info, trajs, cur_renders = evaluate(
            agent=eval_agent,
            env=env,
            task_id=task_id,
            config=config,
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=num_video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
            eval_temperature=FLAGS.eval_temperature,
            eval_gaussian=FLAGS.eval_gaussian,
        )
        del trajs
        renders.extend(cur_renders)
        metric_names = ["success"]
        eval_metrics.update(
            {
                f"evaluation/{task_name}_{k}": v
                for k, v in eval_info.items()
                if k in metric_names
            }
        )
        for k, v in eval_info.items():
            if k in metric_names:
                overall_metrics[k].append(v)
    for k, v in overall_metrics.items():
        eval_metrics[f"evaluation/overall_{k}"] = np.mean(v)
    return (
        eval_metrics,
        renders,
        num_tasks,
        eval_metrics.get("evaluation/overall_success"),
    )


def _prepare_datasets(
    env,
    train_dataset,
    val_dataset,
    dataset_class,
    config,
    dataset_path,
):
    """Preprocess the training/validation pair and wrap it for goal sampling."""
    if FLAGS.clip_dataset_actions:
        eps = 1e-5
        train_dataset = train_dataset.copy(
            add_or_replace={
                "actions": np.clip(train_dataset["actions"], -1 + eps, 1 - eps)
            }
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace={
                    "actions": np.clip(val_dataset["actions"], -1 + eps, 1 - eps)
                }
            )
    if FLAGS.dataset_subtask_split_every > 0:
        train_dataset, split_info = split_trajectories_by_subtasks(
            train_dataset,
            dataset_path,
            FLAGS.dataset_subtask_split_every,
        )
        print(
            "Train dataset split by subtasks: "
            f"every {split_info['split_every']}, "
            f"{split_info['selected_subtask_splits']} internal splits, "
            f"{split_info['original_num_trajectories']} -> "
            f"{split_info['new_num_trajectories']} trajectories, "
            f"length mean/min/max "
            f"{split_info['mean_trajectory_length']:.1f}/"
            f"{split_info['min_trajectory_length']}/"
            f"{split_info['max_trajectory_length']}"
        )
        if val_dataset is not None:
            val_path = (
                dataset_path
                if "-val" in dataset_path
                else dataset_path.replace(".npz", "-val.npz")
            )
            val_dataset, val_split_info = split_trajectories_by_subtasks(
                val_dataset,
                val_path,
                FLAGS.dataset_subtask_split_every,
            )
            print(
                "Validation dataset split by subtasks: "
                f"{val_split_info['selected_subtask_splits']} internal splits, "
                f"{val_split_info['original_num_trajectories']} -> "
                f"{val_split_info['new_num_trajectories']} trajectories"
            )

    if config.get("goal_representation", "full") == "oracle":
        if "antmaze" in FLAGS.env_name or "humanoidmaze" in FLAGS.env_name:
            train_dataset = add_locomaze_oracle_representations(train_dataset)
            if val_dataset is not None:
                val_dataset = add_locomaze_oracle_representations(val_dataset)
        elif "cube" in FLAGS.env_name:
            num_cubes = env.unwrapped._num_cubes
            train_dataset = add_cube_oracle_representations(train_dataset, num_cubes)
            if val_dataset is not None:
                val_dataset = add_cube_oracle_representations(val_dataset, num_cubes)
        elif "puzzle" in FLAGS.env_name:
            num_buttons = env.unwrapped._num_buttons
            train_dataset = add_puzzle_oracle_representations(
                train_dataset, num_buttons
            )
            if val_dataset is not None:
                val_dataset = add_puzzle_oracle_representations(
                    val_dataset, num_buttons
                )
        elif "scene" in FLAGS.env_name:
            num_cubes = env.unwrapped._num_cubes
            num_buttons = env.unwrapped._num_buttons
            train_dataset = add_scene_oracle_representations(
                train_dataset, num_cubes, num_buttons
            )
            if val_dataset is not None:
                val_dataset = add_scene_oracle_representations(
                    val_dataset, num_cubes, num_buttons
                )
        else:
            raise ValueError(
                "--goal_representation=oracle currently supports AntMaze, "
                "HumanoidMaze, Cube, Puzzle, and Scene environments only."
            )

    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)
    return train_dataset, val_dataset


def main(_):
    if FLAGS.dataset_path is None:
        raise ValueError("Provide --dataset_path for the training dataset.")
    dataset_path = os.path.abspath(os.path.expanduser(FLAGS.dataset_path))
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Training dataset does not exist: {dataset_path}")
    if FLAGS.resume_training and (FLAGS.restore_path is None or FLAGS.restore_epoch is None):
        raise ValueError("resume_training requires restore_path and restore_epoch")

    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    exp_name = (
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        + "_"
        + os.path.basename(dataset_path)[:-4]
        + "_"
        + exp_name
    )

    # Set up environment and dataset.
    config = FLAGS.agent
    config["goal_representation"] = FLAGS.goal_representation
    config["precomputed_features"] = bool(FLAGS.precomputed_features)
    env, train_dataset, val_dataset = make_env_and_dataset(
        FLAGS.env_name,
        dataset_path,
        frame_stack=config["frame_stack"],
        precomputed_features=config["precomputed_features"],
        implemtented=True,
    )
    dataset_class = {
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
    }[config["dataset_class"]]
    train_dataset, val_dataset = _prepare_datasets(
        env,
        train_dataset,
        val_dataset,
        dataset_class,
        config,
        dataset_path,
    )


    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    example_obs = example_batch["observations"]
    if config["discrete"]:
        raise ValueError("Only continuous actions are supported.")

    agent_class = agents[config["agent_name"]]

    agent_kwargs = {}
    if FLAGS.learned_encoder_path is not None:
        learned_encoder_run = resolve_run_directory(FLAGS.learned_encoder_path)
        pretrained_encoder, _, _ = _restore_encoder_from_run(
            learned_encoder_run,
            FLAGS.learned_encoder_epoch,
            example_obs,
            example_batch["actions"],
        )
        agent_kwargs["pretrained_encoder"] = pretrained_encoder
    if config["agent_name"] in ("gcfbc", "gciql"):
        agent_kwargs["ex_goals"] = example_batch["actor_goals"]
    elif config["agent_name"] == "hiql":
        agent_kwargs["ex_goals"] = example_batch["high_actor_goals"]
    agent = agent_class.create(
        FLAGS.seed,
        example_obs,
        example_batch["actions"],
        config,
        **agent_kwargs,
    )
    if hasattr(agent, "set_observation_normalizer"):
        agent = agent.set_observation_normalizer(
            train_dataset.observation_statistics,
            train_dataset.norm_method,
        )

    # Restore agent.
    restored_steps = 0
    initial_optimizer_step = int(agent.network.step) if FLAGS.resume_training else 0
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    if FLAGS.resume_training:
        if config["agent_name"] != "gcfbc":
            raise ValueError("resume_training currently supports GCFBC only")
        restored_steps = int(agent.network.step) - initial_optimizer_step
        if not 0 <= restored_steps < FLAGS.train_steps:
            raise ValueError(f"Restored {restored_steps} updates; target must be larger")
        print(f"Resuming {config['agent_name']}: {restored_steps} -> {FLAGS.train_steps} total updates")

    # Optionally load a pre-trained flow model for future augmentation.
    gen_model = None
    gen_model_config = None
    gen_model_flags = None
    if FLAGS.gen_model_path is not None:
        gen_run_dir = resolve_run_directory(FLAGS.gen_model_path)
        gen_model, gen_model_config, gen_model_flags = _restore_flow_model_from_run(
            gen_run_dir,
            FLAGS.gen_model_epoch,
            example_obs,
            example_batch["actions"],
        )
        # print that we restored
        print(
            f"Restored flow model from {gen_run_dir} at epoch {FLAGS.gen_model_epoch}."
        )
        flow_dataset = gen_model_flags.get("dataset_path")
        if flow_dataset and flow_dataset != FLAGS.dataset_path:
            print(
                f"Warning: flow model dataset {flow_dataset} does not match "
                f"current dataset {FLAGS.dataset_path}."
            )
        if gen_model_flags.get("truncate_at_goal", 0):
            raise ValueError("Restoring a goal-truncated generator is not supported.")

        if "maze" in FLAGS.env_name:
            # plot samples to verify working
            gen_model, figs = plot_samples_xy(
                gen_model,
                val_dataset,
                gen_model_config,
                env.unwrapped,
                num_samples_per_state=gen_model_flags["num_eval_samples_per_state"],
                num_plot_states=5,
                plot_actions=gen_model_flags["plot_actions"],
            )
            # save figs to .
            for name, fig in figs.items():
                fig_path = f"gen_model_samples_{name}.png"
                fig.savefig(fig_path)
                plt.close(fig)
                print(f"Saved generative model sample plot to {fig_path}")

    # add gen model config to wandb config
    if gen_model_config is not None:
        config["gen_model_config"] = gen_model_config

    # Setup wandb and saving
    setup_wandb(
        project=FLAGS.project,
        group=FLAGS.run_group,
        name=exp_name,
        entity="stitching-models",
        mode=FLAGS.wandb_mode,
    )
    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    flag_dict["restored_train_steps"] = restored_steps
    flag_dict["additional_train_steps"] = FLAGS.train_steps - restored_steps
    wandb.config.update({"restored_train_steps": restored_steps,
                         "additional_train_steps": FLAGS.train_steps - restored_steps})
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    n_augment = int(config["batch_size"] * FLAGS.gen_prop)
    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()
    successes = []
    for i in tqdm.tqdm(
        range(restored_steps + 1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        # Update agent.
        batch = train_dataset.sample(config["batch_size"])
        if gen_model is not None and n_augment > 0:
            batch, gen_model = gen_augment(
                batch,
                train_dataset,
                n_augment,
                gen_model,
                gen_model_config,
            )
        agent, update_info = agent.update(batch)
        if "stats" in update_info:
            del update_info["stats"]
        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config["batch_size"])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                if "stats" in val_info:
                    del val_info["stats"]
                train_metrics.update(
                    {f"validation/{k}": v for k, v in val_info.items()}
                )
            train_metrics["time/epoch_time"] = (
                time.time() - last_time
            ) / FLAGS.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)


        # Evaluate agent.
        if (FLAGS.eval_at_first_step and i == 1) or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices("cpu")[0])
            else:
                eval_agent = agent
            eval_metrics, renders, num_tasks, overall_success = _evaluate_all_tasks(
                eval_agent=eval_agent,
                env=env,
                config=config,
            )
            if overall_success is not None:
                successes.append(overall_success)
            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics["video"] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)


    # calculate average over last 3 overall_sucess,
    # add to wandb summary
    wandb.run.summary["evaluation/overall_success_3"] = np.mean(successes[-3:])
    wandb.run.summary["evaluation/len_3"] = len(successes[-3:])
    train_logger.close()
    eval_logger.close()


if __name__ == "__main__":
    app.run(main)
