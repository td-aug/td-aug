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

import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.action_prior import inject_online_action_prior as _inject_online_action_prior
from ogbench.env_loader import make_dataset, make_env_and_dataset
from utils.datasets import (
    ChunkTDFlowGCDataset,
    Dataset,
    GCDataset,
    HGCDataset,
    OneStepHistoryGCDataset,
    dict_encode,
    dict_normalize,
    split_trajectories_by_subtasks,
)
from utils.evaluation import (
    evaluate,
)
from utils.flax_utils import restore_agent, save_agent
from utils.gen_evaluation import (
    compute_cube_chunk_emd,
    compute_emd_xy,
    compute_log_likelihood,
    gen_augment,
    plot_chunk_samples_cube,
    plot_chunk_samples_puzzle,
    plot_samples_xy,
    plot_samples_xy_vae,
)
from utils.mchc_discounts import resolve_mchc_discounts
from utils.goal_representations import (
    add_cube_oracle_representations,
    add_locomaze_oracle_representations,
    add_puzzle_oracle_representations,
    add_scene_oracle_representations,
    cube_oracle_representation_jax,
    locomaze_oracle_representation_jax,
    puzzle_oracle_representation_jax,
    scene_oracle_representation_jax,
)
from utils.latent_cache import cache_dataset_observations
from utils.log_utils import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    get_wandb_video,
    setup_wandb,
)
from utils.run_restore import load_run_flags, resolve_run_directory

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "antmaze-large-v0", "Environment name.")
flags.DEFINE_string("dataset_path", None, "Dataset path.")
flags.DEFINE_integer(
    "dataset_subtask_split_every",
    0,
    "Split loaded trajectories after every N subtask switches; 0 disables splitting.",
    lower_bound=0,
)
flags.DEFINE_integer("same_eval", 0, "Whether to use the same dataset for evaluation.")
flags.DEFINE_string("eval_dataset_path", None, "Evaluation dataset path.")
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("wandb_mode", "online", "wandb mode")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")
flags.DEFINE_string("policy_restore_path", None, "Directory containing a GCFBC policy checkpoint.")
flags.DEFINE_integer("policy_restore_epoch", None, "Policy checkpoint filename epoch (global training step). policy_train_steps adds updates after restore.", lower_bound=0)

# Number of steps for Flow training.
flags.DEFINE_integer("train_steps", 500000, "Number of training steps.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", 1000000, "Saving interval.")

flags.DEFINE_integer(
    "policy_train_steps",
    1000000,
    "Number of training steps for policy after flow training.",
)
flags.DEFINE_integer("policy_eval_interval", 100000, "Policy evaluation interval.")
flags.DEFINE_integer(
    "action_prior_train_steps",
    0,
    "State-conditioned action-chunk prior updates before GHM training.",
    lower_bound=0,
)
flags.DEFINE_string(
    "action_prior_restore_path",
    None,
    "Directory or glob containing a saved action-prior checkpoint.",
)
flags.DEFINE_integer(
    "action_prior_restore_epoch",
    None,
    "Checkpoint update to restore from action_prior_restore_path.",
)
flags.DEFINE_integer(
    "action_prior_eval_episodes",
    0,
    "Open-loop evaluation episodes per task after action-prior training.",
    lower_bound=0,
)
flags.DEFINE_integer(
    "action_prior_video_episodes",
    0,
    "Open-loop videos per task recorded after action-prior training.",
    lower_bound=0,
)
flags.DEFINE_list(
    "action_prior_eval_steps",
    [],
    "Optional comma-separated action-prior update counts to evaluate/checkpoint.",
)
flags.DEFINE_list(
    "action_prior_eval_restore_epochs",
    [],
    "Optional saved action-prior epochs to evaluate sequentially without training.",
)
flags.DEFINE_integer(
    "action_prior_eval_interval",
    0,
    "Action-prior evaluation interval; 0 evaluates only explicit steps and the end.",
    lower_bound=0,
)
flags.DEFINE_bool(
    "action_prior_only",
    False,
    "Stop after training and evaluating the state-conditioned action prior.",
)
flags.DEFINE_float(
    "gen_prop",
    0.5,
    "Proportion of batch that comes from generative model augmentation.",
    lower_bound=0.0, upper_bound=1.0,
)
flags.DEFINE_enum(
    "augmentation_mode", "online", ["online", "onestep_ram_cache"],
    "Online goal generation (default) or one frozen one-step trajectory per dataset row in CPU RAM.",
)
flags.DEFINE_integer(
    "rollout_cache_batch_size", 2048,
    "Number of source trajectories generated together for onestep_ram_cache.",
    lower_bound=1,
)
flags.DEFINE_string(
    "learned_encoder_path", None, "Path to checkpoint of learned encoder to use."
)
flags.DEFINE_integer(
    "learned_encoder_epoch", None, "Epoch of checkpoint of learned encoder to use."
)
flags.DEFINE_integer(
    "encoder_train_steps",
    0,
    "Number of training steps for encoder when training from scratch.",
)
flags.DEFINE_integer(
    "encoder_eval_interval",
    5000,
    "Evaluation interval for encoder when training from scratch.",
)
flags.DEFINE_integer(
    "precomputed_features", 0, "Whether the dataset contains precomputed features."
)

flags.DEFINE_integer("eval_tasks", None, "Number of tasks to evaluate (None for all).")
flags.DEFINE_integer("eval_episodes", 50, "Number of episodes for each task.")
flags.DEFINE_float("eval_temperature", 0, "Actor temperature for evaluation.")
flags.DEFINE_float("eval_gaussian", None, "Action Gaussian noise for evaluation.")
flags.DEFINE_integer(
    "num_eval_samples_per_state",
    256,
    "Number of samples to generate per state for EMD/V-MSE/Log-likelihood.",
)
flags.DEFINE_integer(
    "num_eval_states", 50, "Number of states to use for evaluation metrics."
)
flags.DEFINE_integer("compute_emd_xy", 0, "Whether to compute EMD_XY during evaluation.")
flags.DEFINE_integer(
    "compute_cube_emd",
    0,
    "Whether to compute per-source raw-state and cube-position EMD.",
)
flags.DEFINE_enum(
    "cube_eval_source_mode",
    "random",
    ["random", "initial"],
    "Source-state selection for external Cube evaluation diagnostics.",
)
flags.DEFINE_integer("num_emd_trajectories", 128, "Number of source trajectories for EMD_XY.")
flags.DEFINE_integer(
    "num_emd_samples_per_state",
    128,
    "Number of future samples per source state for EMD_XY.",
)
flags.DEFINE_float(
    "emd_discount",
    -1.0,
    "Discount used for geometric future-state sampling in EMD_XY; negative uses model discount.",
)
flags.DEFINE_float(
    "metric_discount",
    -1.0,
    "Discount used for geometric future-state sampling in log-likelihood metrics, and as the EMD_XY default; negative uses model discount.",
)
flags.DEFINE_float("emd_epsilon", 1e-2, "Sinkhorn epsilon for EMD_XY.")
flags.DEFINE_integer(
    "emd_sinkhorn_iters", 100, "Number of Sinkhorn iterations for EMD_XY."
)
flags.DEFINE_integer(
    "encoder_num_eval_samples_per_state",
    4,
    "Number of samples to generate per state for EMD/V-MSE/Log-likelihood.",
)
flags.DEFINE_integer(
    "encoder_num_eval_states", 5, "Number of states to use for evaluation metrics."
)
flags.DEFINE_integer(
    "cache_encoder_latents",
    0,
    "Whether to cache full-dataset encoder latents for flow and policy training.",
)
flags.DEFINE_integer(
    "latent_cache_batch_size",
    4096,
    "Batch size used to encode the full observation table for latent caching.",
)

flags.DEFINE_integer("plot_samples", 1, "Whether to plot samples in the evaluation.")
flags.DEFINE_integer(
    "plot_actions", 0, "Whether to plot actions in the evaluation graphs."
)
flags.DEFINE_integer("video_episodes", 1, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")
flags.DEFINE_integer("eval_on_cpu", 0, "Whether to evaluate on CPU.")

flags.DEFINE_string("project", "comb-gen-gen", "WandB project name.")

config_flags.DEFINE_config_file("agent", "agents/tdflow.py", lock_config=False)
config_flags.DEFINE_config_file("policy", None, lock_config=False)
config_flags.DEFINE_config_file("encoder", None, lock_config=False)
config_flags.DEFINE_config_file("action_prior", None, lock_config=False)


def _evaluate_action_prior(action_prior, env, config, num_tasks):
    """Evaluate the non-goal-conditioned prior from each task start state."""
    metrics = {}
    successes = []
    renders = []
    task_infos = (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, "task_infos")
        else env.task_infos
    )
    for task_id in tqdm.trange(1, num_tasks + 1, desc="Action-prior evaluation"):
        task_name = task_infos[task_id - 1]["task_name"]
        info, trajectories, task_renders = evaluate(
            agent=action_prior,
            env=env,
            task_id=task_id,
            config=config,
            num_eval_episodes=FLAGS.action_prior_eval_episodes,
            num_video_episodes=FLAGS.action_prior_video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
            eval_temperature=FLAGS.eval_temperature,
            eval_gaussian=FLAGS.eval_gaussian,
        )
        if "success" in info:
            success = info["success"]
            metrics[f"action_prior/evaluation/{task_name}_success"] = success
            successes.append(success)
        if trajectories:
            metrics[f"action_prior/evaluation/{task_name}_return"] = np.mean(
                [np.sum(trajectory["reward"]) for trajectory in trajectories]
            )
            metrics[f"action_prior/evaluation/{task_name}_length"] = np.mean(
                [len(trajectory["reward"]) for trajectory in trajectories]
            )
        renders.extend(task_renders)
    if successes:
        metrics["action_prior/evaluation/overall_success"] = np.mean(successes)
    if renders:
        metrics["action_prior/video"] = get_wandb_video(
            renders=renders, n_cols=num_tasks
        )
    return metrics




def _normalize_optional_path(path):
    if path is None:
        return None
    if isinstance(path, str):
        path = path.strip()
        if path == "":
            return None
    return path




def _restore_encoder_from_run(run_dir, epoch, example_obs, example_act):
    """Instantiate and restore a flow model from a saved run directory."""
    if epoch is None:
        raise ValueError(
            "learned_encoder_epoch must be set when providing encoder_path."
        )
    flag_dict = load_run_flags(run_dir)
    config = copy.deepcopy(flag_dict["encoder"])
    agent_name = config.get("agent_name")
    if agent_name not in agents:
        raise ValueError(f"Unknown flow agent {agent_name} in {run_dir}")
    encoder_cls = agents[agent_name]
    seed = flag_dict.get("seed", 0)
    encoder_model = encoder_cls.create(
        seed,
        example_obs,
        example_act,
        config,
    )
    encoder_model = restore_agent(encoder_model, run_dir, epoch)
    return encoder_model, config, flag_dict


def _validate_mc_hc_discount_config(config_model):
    """Validate the fixed local and final MC-HC discounts."""
    if config_model.get("loss_type") != "MC_HC_TD2":
        return
    discount, final_discount = resolve_mchc_discounts(config_model)
    assert 0.0 <= discount < 1.0, "MC_HC_TD2 requires 0 <= discount < 1."
    assert 0.0 <= final_discount < 1.0, "MC_HC_TD2 requires 0 <= final_discount < 1."
    assert final_discount >= discount, "MC_HC_TD2 requires final_discount >= discount."


def _set_dataset_discount(datasets, discount):
    for dataset in datasets:
        if dataset is not None and hasattr(dataset, "set_discount"):
            dataset.set_discount(discount)


def _set_dataset_loss_type(datasets, loss_type):
    for dataset in datasets:
        if dataset is not None and hasattr(dataset, "config"):
            dataset.config["loss_type"] = loss_type


def _get_dataset_discount(dataset):
    if dataset is None or not hasattr(dataset, "config"):
        return None
    return dataset.config.get("discount", None)


def _positive_steps(values, flag_name):
    """Parse a comma-separated list of positive checkpoint/update counts."""
    try:
        steps = [int(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"--{flag_name} must contain positive integers.") from exc
    if any(step <= 0 for step in steps):
        raise ValueError(f"--{flag_name} must contain positive integers.")
    return steps


def main(_):
    policy_restore_path = _normalize_optional_path(FLAGS.policy_restore_path)
    if (policy_restore_path is None) != (FLAGS.policy_restore_epoch is None):
        raise ValueError("Provide both --policy_restore_path and --policy_restore_epoch.")
    action_prior_restore_path = _normalize_optional_path(
        FLAGS.action_prior_restore_path
    )

    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    exp_name = (
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        + "_"
        + FLAGS.dataset_path[:-4].split("/")[-1]
        + "_"
        + exp_name
    )
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
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config_model = FLAGS.agent
    config_policy = FLAGS.policy
    config_encoder = FLAGS.encoder
    config_action_prior = FLAGS.action_prior
    if FLAGS.augmentation_mode == "onestep_ram_cache":
        if config_model["agent_name"] != "onestep" or config_policy is None:
            raise ValueError("onestep_ram_cache requires agents/onestep.py and a policy")
        if FLAGS.gen_prop == 0:
            raise ValueError("RAM caching requires gen_prop > 0.")
        if (config_policy["agent_name"] != "gcfbc"
                or config_policy.get("action_chunk_len", 1) != 1
                or config_policy.get("observation_history_size", 1) != 1
                or config_policy.get("goal_representation", "full") != "full"):
            raise ValueError("RAM caching currently requires full-goal GCFBC with H1 policy states/actions")
    action_prior_eval_steps = set(_positive_steps(
        FLAGS.action_prior_eval_steps, "action_prior_eval_steps"
    ))
    if any(
        step > FLAGS.action_prior_train_steps for step in action_prior_eval_steps
    ):
        raise ValueError(
            "--action_prior_eval_steps cannot exceed --action_prior_train_steps."
        )
    action_prior_eval_restore_epochs = _positive_steps(
        FLAGS.action_prior_eval_restore_epochs, "action_prior_eval_restore_epochs"
    )
    if len(set(action_prior_eval_restore_epochs)) != len(
        action_prior_eval_restore_epochs
    ):
        raise ValueError(
            "--action_prior_eval_restore_epochs must not contain duplicates."
        )
    uses_action_prior = (
        config_model.get("bootstrap_continuation_mode", "recorded")
        == "policy_prior"
    )
    if uses_action_prior and config_action_prior is None:
        raise ValueError(
            "bootstrap_continuation_mode='policy_prior' requires "
            "--action_prior=agents/gcfbc.py."
        )
    if (
        config_action_prior is not None
        and not uses_action_prior
        and not FLAGS.action_prior_only
    ):
        raise ValueError(
            "--action_prior requires agent.bootstrap_continuation_mode=policy_prior."
        )
    if (action_prior_restore_path is None) != (FLAGS.action_prior_restore_epoch is None):
        raise ValueError("Provide both --action_prior_restore_path and --action_prior_restore_epoch.")
    if (
        config_action_prior is not None
        and FLAGS.action_prior_train_steps <= 0
        and action_prior_restore_path is None
    ):
        raise ValueError(
            "An action prior requires training steps or a restore checkpoint."
        )
    if FLAGS.action_prior_only:
        if config_action_prior is None:
            raise ValueError("--action_prior_only requires --action_prior.")
        if FLAGS.train_steps != 0 or FLAGS.policy_train_steps != 0:
            raise ValueError(
                "--action_prior_only requires --train_steps=0 and "
                "--policy_train_steps=0."
            )
        if config_policy is not None:
            raise ValueError("--action_prior_only does not accept --policy.")
    if action_prior_eval_restore_epochs:
        if not FLAGS.action_prior_only:
            raise ValueError(
                "--action_prior_eval_restore_epochs requires --action_prior_only=True."
            )
        if FLAGS.action_prior_train_steps != 0:
            raise ValueError(
                "--action_prior_eval_restore_epochs requires "
                "--action_prior_train_steps=0."
            )
        if action_prior_restore_path is None:
            raise ValueError(
                "--action_prior_eval_restore_epochs requires "
                "--action_prior_restore_path."
            )
        if (
            FLAGS.action_prior_eval_episodes == 0
            and FLAGS.action_prior_video_episodes == 0
        ):
            raise ValueError(
                "--action_prior_eval_restore_epochs requires rollout episodes or "
                "video episodes."
            )
    if (uses_action_prior and config_model["agent_name"] == "tdflow"
            and config_model["loss_type"] != "MC_HC_TD2"):
        raise ValueError("tdflow action-prior bootstrapping requires MC_HC_TD2.")
    config_model["precomputed_features"] = bool(FLAGS.precomputed_features)
    if config_policy is not None:
        config_policy["precomputed_features"] = bool(FLAGS.precomputed_features)
    if config_action_prior is not None:
        config_action_prior["precomputed_features"] = False
        config_action_prior["encode_observations"] = False
        config_action_prior["frame_stack"] = None

    env, train_dataset, val_dataset = make_env_and_dataset(
        FLAGS.env_name,
        FLAGS.dataset_path,
        frame_stack=config_model["frame_stack"],
        precomputed_features=config_model["precomputed_features"],
        implemtented=True,
        same_eval=bool(FLAGS.same_eval),
    )
    if FLAGS.dataset_subtask_split_every > 0:
        train_dataset, split_info = split_trajectories_by_subtasks(
            train_dataset,
            FLAGS.dataset_path,
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
                FLAGS.dataset_path
                if "-val" in FLAGS.dataset_path
                else FLAGS.dataset_path.replace(".npz", "-val.npz")
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
    eval_dataset = None
    if FLAGS.eval_dataset_path is not None:
        eval_dataset = make_dataset(
            FLAGS.env_name,
            FLAGS.eval_dataset_path,
            precomputed_features=config_model["precomputed_features"],
        )


    if config_action_prior is not None:
        if config_action_prior["agent_name"] != "gcfbc":
            raise ValueError("The action prior currently requires agent_name='gcfbc'.")
        if not config_action_prior.get("single_input_to_actor", False):
            raise ValueError("The action prior requires single_input_to_actor=True.")
        if config_action_prior.get("encoder", None) is not None:
            raise ValueError("The action prior requires encoder=None.")
        if config_action_prior.get("normalize_observations", False):
            raise ValueError(
                "The action prior expects raw observations."
            )
        if FLAGS.precomputed_features or config_encoder is not None:
            raise ValueError(
                "The action prior supports raw states without encoders."
            )
        if int(config_model.get("history_size", 1)) != 1:
            raise ValueError("The action prior currently requires agent.history_size=1.")
        if int(config_action_prior.get("observation_history_size", 1)) != 1:
            raise ValueError(
                "The action prior currently requires observation_history_size=1."
            )
        if int(config_action_prior["action_chunk_len"]) != int(
            config_model.get("action_chunk_size", 1)
        ):
            raise ValueError(
                "action_prior.action_chunk_len must equal agent.action_chunk_size."
            )
    generated_goal_transform = None
    oracle_policy_base_datasets = None
    num_cubes = None
    if (
        config_policy is not None
        and config_policy.get("goal_representation", "full") == "oracle"
    ):
        if "antmaze" in FLAGS.env_name or "humanoidmaze" in FLAGS.env_name:
            oracle_policy_base_datasets = tuple(
                (
                    add_locomaze_oracle_representations(dataset)
                    if dataset is not None
                    else None
                )
                for dataset in (train_dataset, val_dataset, eval_dataset)
            )
            generated_goal_transform = locomaze_oracle_representation_jax
        elif "cube-" in FLAGS.env_name:
            if num_cubes is None:
                num_cubes = int(env.unwrapped._num_cubes)
            oracle_policy_base_datasets = tuple(
                (
                    add_cube_oracle_representations(dataset, num_cubes)
                    if dataset is not None
                    else None
                )
                for dataset in (train_dataset, val_dataset, eval_dataset)
            )
            generated_goal_transform = lambda goals: (
                cube_oracle_representation_jax(goals, num_cubes)
            )
        elif "puzzle-" in FLAGS.env_name:
            num_buttons = int(env.unwrapped._num_buttons)
            oracle_policy_base_datasets = tuple(
                (
                    add_puzzle_oracle_representations(dataset, num_buttons)
                    if dataset is not None
                    else None
                )
                for dataset in (train_dataset, val_dataset, eval_dataset)
            )
            generated_goal_transform = lambda goals: (
                puzzle_oracle_representation_jax(goals, num_buttons)
            )
        elif "scene" in FLAGS.env_name:
            num_cubes = int(env.unwrapped._num_cubes)
            num_buttons = int(env.unwrapped._num_buttons)
            oracle_policy_base_datasets = tuple(
                (
                    add_scene_oracle_representations(
                        dataset, num_cubes, num_buttons
                    )
                    if dataset is not None
                    else None
                )
                for dataset in (train_dataset, val_dataset, eval_dataset)
            )
            generated_goal_transform = lambda goals: (
                scene_oracle_representation_jax(
                    goals, num_cubes, num_buttons
                )
            )
        else:
            raise ValueError(
                "policy.goal_representation=oracle currently supports AntMaze, "
                "HumanoidMaze, Cube, Puzzle, and Scene environments only."
            )


    if config_policy:
        data_config = copy.copy(config_policy)
        if "norm_method" in config_policy:
            assert config_policy["norm_method"] == config_model["norm_method"], (
                "Policy and model norm method param need match, currently",
            )
        else:
            config_policy["norm_method"] = config_model["norm_method"]
        if config_model["loss_type"] == "MC":
            assert (
                config_policy["value_geom_sample"] == config_model["value_geom_sample"]
            ), ("Policy and model value_geom_sample param need match, currently",)
        data_config["normalize_observations"] = config_model["normalize_observations"]
        data_config["norm_method"] = config_model["norm_method"]
        data_config["discount"] = config_model["discount"]
        if "final_discount" in config_model:
            data_config["final_discount"] = config_model["final_discount"]
        data_config["loss_type"] = config_model["loss_type"]
        if config_model["agent_name"] == "onestep":
            data_config["dataset_class"] = config_model["dataset_class"]
            data_config["history_size"] = config_model.get("history_size", 1)
        if config_model["agent_name"] == "chunk_tdflow":
            data_config["dataset_class"] = config_model["dataset_class"]
            data_config["action_chunk_size"] = config_model["action_chunk_size"]
            for key in (
                "bootstrap_stride",
                "mchc_sampling_unit",
                "bootstrap_continuation_mode",
            ):
                if key in config_model:
                    data_config[key] = config_model[key]
    else:
        data_config = config_model

    # We create dataset w/ policy_config to ensure compatibility with policy training later on.
    # This is because flow training on uses obs, action, next_obs but policy training cares about goal dist
    dataset_class = {
        "ChunkTDFlowGCDataset": ChunkTDFlowGCDataset,
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
        "OneStepHistoryGCDataset": OneStepHistoryGCDataset,
    }[data_config["dataset_class"]]
    train_dataset = dataset_class(Dataset.create(**train_dataset), data_config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), data_config)
    if eval_dataset is not None:
        eval_dataset = dataset_class(Dataset.create(**eval_dataset), data_config)
    action_prior_datasets = None
    if config_action_prior is not None:
        action_prior_data_config = copy.copy(config_action_prior)
        action_prior_data_config["normalize_observations"] = False
        action_prior_datasets = tuple(
            (
                GCDataset(
                    wrapped_dataset.dataset,
                    copy.copy(action_prior_data_config),
                )
                if wrapped_dataset is not None
                else None
            )
            for wrapped_dataset in (train_dataset, val_dataset, eval_dataset)
        )
    oracle_policy_datasets = None
    if oracle_policy_base_datasets is not None:
        policy_data_config = copy.copy(config_policy)
        policy_data_config["normalize_observations"] = False
        policy_data_config["norm_method"] = data_config["norm_method"]
        oracle_policy_datasets = tuple(
            (
                GCDataset(Dataset.create(**dataset), copy.copy(policy_data_config))
                if dataset is not None
                else None
            )
            for dataset in oracle_policy_base_datasets
        )
        _set_dataset_discount(oracle_policy_datasets, config_policy["discount"])
    _set_dataset_discount(
        (train_dataset, val_dataset, eval_dataset), config_model["discount"]
    )
    print(
        "[main_gen] dataset discount initialized for model stage: "
        f"train={_get_dataset_discount(train_dataset)}, "
        f"val={_get_dataset_discount(val_dataset)}, "
        f"eval={_get_dataset_discount(eval_dataset)}, "
        f"agent_discount={config_model['discount']}, "
        f"policy_discount={config_policy['discount'] if config_policy else None}"
    )
    _validate_mc_hc_discount_config(config_model)
    if config_model.get("loss_type") == "MC_HC_TD2":
        effective_local_discount, effective_final_discount = (
            resolve_mchc_discounts(config_model)
        )
        discount_metrics = {
            "config/mchc_sampling_unit": config_model.get(
                "mchc_sampling_unit", "primitive"
            ),
            "config/mchc_raw_local_discount": config_model["discount"],
            "config/mchc_raw_final_discount": config_model["final_discount"],
            "config/mchc_effective_local_discount": effective_local_discount,
            "config/mchc_effective_final_discount": effective_final_discount,
        }
        print("[main_gen] MCHC discounts:", discount_metrics)
        wandb.log(discount_metrics, step=0)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    policy_example_batch = (
        oracle_policy_datasets[0].sample(1)
        if oracle_policy_datasets is not None
        else example_batch
    )
    action_prior_example_batch = (
        action_prior_datasets[0].sample(1)
        if action_prior_datasets is not None
        else None
    )
    if config_model["discrete"]:
        raise ValueError("Only continuous actions are supported.")

    example_obs = example_batch["observations"]

    model_kwargs = {}
    policy_kwargs = {}
    pretrained_encoder = None
    encode_observations = False
    pretrained_decoder = None
    enc_obs = None
    if FLAGS.learned_encoder_path is not None:
        assert (
            not config_encoder
            # encoder_train_steps == 0
        ), "Encoder config should not be set when using pretrained encoder."
        enc_obs = example_obs
        encoder = _restore_encoder_from_run(
            FLAGS.learned_encoder_path,
            FLAGS.learned_encoder_epoch,
            enc_obs,
            example_batch["actions"],
        )[0]
        if hasattr(encoder, "decode"):
            pretrained_decoder = encoder

        model_kwargs["pretrained_encoder"] = encoder
        policy_kwargs["pretrained_encoder"] = encoder
        pretrained_encoder = encoder
        assert not config_model[
            "normalize_observations"
        ], "Do not use observation normalization with pretrained encoder, encoder should handle"
        if config_policy:
            assert (
                config_policy["precomputed_features"]
                == config_model["precomputed_features"]
            ), "Policy and model precomputed_features must match."
        encode_observations = not config_model["precomputed_features"]
        config_model["encode_observations"] = encode_observations
        if config_policy is not None:
            config_policy["encode_observations"] = encode_observations
    elif config_encoder is not None:
        encoder_cls = agents[config_encoder["agent_name"]]
        assert not config_encoder[
            "normalize_observations"
        ], "currently not supported, expected to internal normalize"
        encoder = encoder_cls.create(
            FLAGS.seed,
            example_obs,
            example_batch["actions"],
            config_encoder,
        )
        pretrained_encoder = encoder
        if hasattr(encoder, "decode"):
            pretrained_decoder = encoder
        model_kwargs["pretrained_encoder"] = pretrained_encoder
        policy_kwargs["pretrained_encoder"] = pretrained_encoder
        config_model["encode_observations"] = True
        if config_policy is not None:
            config_policy["encode_observations"] = True

        assert not config_model[
            "normalize_observations"
        ], "Do not use observation normalization with pretrained encoder, encoder should handle"
        encode_observations = True

    model_cls = agents[config_model["agent_name"]]
    model_example_actions = (
        example_batch.get("condition_actions", example_batch["actions"])
    )
    model = model_cls.create(
        FLAGS.seed,
        example_obs,
        model_example_actions,
        config_model,
        **model_kwargs,
    )


    action_prior = None
    action_prior_save_dir = None
    if config_action_prior is not None:
        action_prior_cls = agents[config_action_prior["agent_name"]]
        action_prior = action_prior_cls.create(
            FLAGS.seed + 1_000_003,
            action_prior_example_batch["observations"],
            action_prior_example_batch["actions"],
            config_action_prior,
        )
        if action_prior_restore_path is not None:
            action_prior = restore_agent(
                action_prior,
                action_prior_restore_path,
                FLAGS.action_prior_restore_epoch,
            )
        action_prior_save_dir = os.path.join(FLAGS.save_dir, "action_prior")
        os.makedirs(action_prior_save_dir, exist_ok=True)
        wandb.log(
            {
                "config/action_prior_train_steps": FLAGS.action_prior_train_steps,
                "config/action_prior_restore_path": action_prior_restore_path,
                "config/action_prior_restore_epoch": FLAGS.action_prior_restore_epoch,
                "config/action_prior_eval_steps": sorted(action_prior_eval_steps),
                "config/action_prior_eval_restore_epochs": (
                    action_prior_eval_restore_epochs
                ),
                "config/action_prior_eval_interval": (
                    FLAGS.action_prior_eval_interval
                ),
                "config/action_prior_only": int(FLAGS.action_prior_only),
                "config/action_prior_action_chunk_len": config_action_prior[
                    "action_chunk_len"
                ],
            },
            step=0,
        )


    policy_cls = None
    policy_model = None
    if config_policy is not None:
        policy_cls = agents[config_policy["agent_name"]]
        if config_policy["agent_name"] == "gcfbc":
            policy_kwargs["ex_goals"] = policy_example_batch["actor_goals"]
        policy_model = policy_cls.create(
            FLAGS.seed,
            policy_example_batch["observations"],
            policy_example_batch["actions"],
            config_policy,
            **policy_kwargs,
        )

    # Restore model.
    if FLAGS.restore_path is not None:
        model = restore_agent(model, FLAGS.restore_path, FLAGS.restore_epoch)
        print(
            f'Restored agent from {os.path.join(FLAGS.restore_path, f"params_{FLAGS.restore_epoch}.pkl")}'
        )

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()
    successes = []

    # Train encoder from scratch if specified
    if config_encoder is not None and FLAGS.encoder_train_steps > 0:
        encoder_train_dataset = (
            train_dataset.dataset if FLAGS.cache_encoder_latents else train_dataset
        )
        encoder_val_dataset = (
            val_dataset.dataset
            if FLAGS.cache_encoder_latents and val_dataset is not None
            else val_dataset
        )
        for i in tqdm.tqdm(
            range(
                1,
                1 + FLAGS.encoder_train_steps,
            ),
            smoothing=0.1,
            dynamic_ncols=True,
            desc="Encoder training",
        ):
            rel_step = i
            batch = encoder_train_dataset.sample(config_encoder["batch_size"])
            if config_encoder["normalize_observations"]:
                batch = dict_normalize(
                    batch,
                    train_dataset.observation_statistics,
                    train_dataset.norm_method,
                )
            pretrained_encoder, update_info = pretrained_encoder.update(batch)
            if "stats" in update_info:
                del update_info["stats"]
            # Log metrics.
            if i % FLAGS.log_interval == 0:
                train_metrics = {f"training/{k}": v for k, v in update_info.items()}
                if encoder_val_dataset is not None:
                    val_batch = encoder_val_dataset.sample(config_encoder["batch_size"])
                    if config_encoder["normalize_observations"]:
                        val_batch = dict_normalize(
                            val_batch,
                            val_dataset.observation_statistics,
                            val_dataset.norm_method,
                        )
                    pretrained_encoder, val_loss, val_info = (
                        pretrained_encoder.compute_loss(batch=val_batch)
                    )
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
                # wandb.log(train_metrics, step=i)
                wandb.log({**train_metrics, "rel_step": rel_step}, step=i)
                train_logger.log(train_metrics, step=i)
            eval_logs = None
            if i == 1 or i % FLAGS.encoder_eval_interval == 0:
                if (
                    "vae" in config_encoder["agent_name"]
                    or "fae" in config_encoder["agent_name"]
                ):
                    # debug = True if i == 1000 else False
                    debug = False
                    figs = plot_samples_xy_vae(
                        pretrained_encoder,
                        val_dataset,
                        config_encoder,
                        env.unwrapped,
                        num_samples_per_state=FLAGS.encoder_num_eval_samples_per_state,
                        num_plot_states=FLAGS.encoder_num_eval_states,
                        plot_images=example_obs.ndim == 4,
                        debug=debug,
                    )
                else:
                    figs = dict()

                for name, fig in figs.items():
                    if fig is None:
                        continue
                    wandb.log(
                        {
                            f"evaluation/val/sample_xy/{name}": wandb.Image(fig),
                            "rel_step": rel_step,
                        },
                        step=i,
                    )

                plt.close("all")
                if FLAGS.eval_episodes > 0 and hasattr(
                    pretrained_encoder, "sample_actions"
                ):
                    if FLAGS.eval_on_cpu:
                        eval_agent = jax.device_put(
                            pretrained_encoder, device=jax.devices("cpu")[0]
                        )
                    else:
                        eval_agent = pretrained_encoder
                    encoder_eval_metrics = {}
                    overall_metrics = defaultdict(list)
                    task_infos = (
                        env.unwrapped.task_infos
                        if hasattr(env.unwrapped, "task_infos")
                        else env.task_infos
                    )
                    num_tasks = (
                        FLAGS.eval_tasks
                        if FLAGS.eval_tasks is not None
                        else len(task_infos)
                    )
                    for task_id in tqdm.trange(1, num_tasks + 1):
                        task_name = task_infos[task_id - 1]["task_name"]
                        task_eval_info, _, _ = evaluate(
                            agent=eval_agent,
                            env=env,
                            task_id=task_id,
                            config=config_encoder,
                            num_eval_episodes=FLAGS.eval_episodes,
                            num_video_episodes=0,
                            video_frame_skip=FLAGS.video_frame_skip,
                            eval_temperature=FLAGS.eval_temperature,
                            eval_gaussian=FLAGS.eval_gaussian,
                        )
                        metric_names = ["success"]
                        encoder_eval_metrics.update(
                            {
                                f"encoder_evaluation/{task_name}_{k}": v
                                for k, v in task_eval_info.items()
                                if k in metric_names
                            }
                        )
                        for k, v in task_eval_info.items():
                            if k in metric_names:
                                overall_metrics[k].append(v)
                    for k, v in overall_metrics.items():
                        encoder_eval_metrics[f"encoder_evaluation/overall_{k}"] = np.mean(v)
                    if eval_logs is None:
                        eval_logs = {}
                    eval_logs.update(encoder_eval_metrics)

                if eval_logs is not None:
                    wandb.log({**eval_logs, "rel_step": rel_step}, step=i)
                    eval_logger.log(eval_logs, step=i)

            if i % FLAGS.save_interval == 0:
                save_agent(pretrained_encoder, FLAGS.save_dir, i)
        if FLAGS.encoder_train_steps % FLAGS.save_interval != 0:
            save_agent(pretrained_encoder, FLAGS.save_dir, FLAGS.encoder_train_steps)

    # after training pretraining encoder
    if policy_model is not None and pretrained_encoder is not None:
        policy_model = policy_model.replace(pretrained_encoder=pretrained_encoder)
    if hasattr(pretrained_encoder, "decode"):
        pretrained_decoder = pretrained_encoder

    model_train_dataset = train_dataset
    model_val_dataset = val_dataset
    model_eval_dataset = eval_dataset
    if oracle_policy_datasets is not None:
        policy_train_dataset, policy_val_dataset, policy_eval_dataset = (
            oracle_policy_datasets
        )
    else:
        policy_train_dataset = train_dataset
        policy_val_dataset = val_dataset
        policy_eval_dataset = eval_dataset
    model_batches_are_cached_latents = False
    policy_batches_are_cached_latents = False
    if FLAGS.cache_encoder_latents:
        if not encode_observations or pretrained_encoder is None:
            raise ValueError(
                "cache_encoder_latents requires an active learned/pretrained encoder."
            )
        if float(data_config.get("p_aug", 0.0) or 0.0) > 0.0:
            raise ValueError(
                "cache_encoder_latents requires p_aug=0 because image augmentation "
                "must happen before encoding."
            )

        raw_wrapped_datasets = (train_dataset, val_dataset, eval_dataset)
        cached_bases = []
        for wrapped_dataset in raw_wrapped_datasets:
            if wrapped_dataset is None:
                cached_bases.append(None)
                continue
            cached_bases.append(
                cache_dataset_observations(
                    wrapped_dataset.dataset,
                    pretrained_encoder,
                    batch_size=FLAGS.latent_cache_batch_size,
                )
            )

        model_datasets = tuple(
            (
                dataset_class(base_dataset, copy.copy(data_config))
                if base_dataset is not None
                else None
            )
            for base_dataset in cached_bases
        )
        model_train_dataset, model_val_dataset, model_eval_dataset = model_datasets
        _set_dataset_discount(model_datasets, config_model["discount"])

        if config_policy is not None:
            policy_data_config = copy.copy(config_policy)
            policy_data_config["normalize_observations"] = False
            policy_data_config["norm_method"] = data_config["norm_method"]
            policy_datasets = tuple(
                (
                    GCDataset(base_dataset, copy.copy(policy_data_config))
                    if base_dataset is not None
                    else None
                )
                for base_dataset in cached_bases
            )
            (
                policy_train_dataset,
                policy_val_dataset,
                policy_eval_dataset,
            ) = policy_datasets
            _set_dataset_discount(policy_datasets, config_policy["discount"])

        model_batches_are_cached_latents = True
        policy_batches_are_cached_latents = config_policy is not None
        latent_shape = model_train_dataset.dataset["observations"].shape
        latent_bytes = model_train_dataset.dataset["observations"].nbytes
        print(
            "[main_gen] cached encoder latents enabled: "
            f"train_shape={latent_shape}, train_size_mib={latent_bytes / 2**20:.1f}"
        )
        wandb.log(
            {
                "config/cache_encoder_latents": 1,
                "config/latent_cache_train_size_mib": latent_bytes / 2**20,
            },
            step=FLAGS.encoder_train_steps,
        )

    pre_model_steps = FLAGS.encoder_train_steps
    action_prior_sample_rng = None
    if action_prior is not None:
        prior_train_dataset, prior_val_dataset, _ = action_prior_datasets
        prior_stage_start = pre_model_steps + 1
        prior_eval_steps = set(action_prior_eval_steps)
        prior_eval_steps.add(FLAGS.action_prior_train_steps)
        prior_eval_enabled = (
            FLAGS.action_prior_eval_episodes > 0
            or FLAGS.action_prior_video_episodes > 0
        )
        task_infos = (
            env.unwrapped.task_infos
            if hasattr(env.unwrapped, "task_infos")
            else env.task_infos
        )
        prior_num_tasks = (
            FLAGS.eval_tasks
            if FLAGS.eval_tasks is not None
            else len(task_infos)
        )
        if action_prior_eval_restore_epochs:
            for restore_epoch in action_prior_eval_restore_epochs:
                action_prior = restore_agent(
                    action_prior,
                    action_prior_restore_path,
                    restore_epoch,
                )
                prior_eval_metrics = _evaluate_action_prior(
                    action_prior,
                    env,
                    config_action_prior,
                    prior_num_tasks,
                )
                prior_eval_metrics["action_prior/evaluation/checkpoint"] = (
                    restore_epoch
                )
                wandb.log(
                    {**prior_eval_metrics, "rel_step": restore_epoch},
                    step=restore_epoch,
                )
                eval_logger.log(prior_eval_metrics, step=restore_epoch)

            train_logger.close()
            eval_logger.close()
            env.close()
            return

        for i in tqdm.tqdm(
            range(
                prior_stage_start,
                prior_stage_start + FLAGS.action_prior_train_steps,
            ),
            smoothing=0.1,
            dynamic_ncols=True,
            desc="Action-prior training",
        ):
            rel_step = i - pre_model_steps
            prior_batch = prior_train_dataset.sample(
                config_action_prior["batch_size"]
            )
            action_prior, prior_update_info = action_prior.update(prior_batch)
            if rel_step % FLAGS.log_interval == 0:
                prior_metrics = {
                    f"action_prior/training/{key}": value
                    for key, value in prior_update_info.items()
                }
                if prior_val_dataset is not None:
                    prior_val_batch = prior_val_dataset.sample(
                        config_action_prior["batch_size"]
                    )
                    _, prior_val_info = action_prior.total_loss(
                        prior_val_batch, action_prior.network.params
                    )
                    prior_metrics.update(
                        {
                            f"action_prior/validation/{key}": value
                            for key, value in prior_val_info.items()
                        }
                    )
                prior_metrics["time/total_time"] = time.time() - first_time
                wandb.log({**prior_metrics, "rel_step": rel_step}, step=i)
                train_logger.log(prior_metrics, step=i)

            periodic_prior_eval = (
                FLAGS.action_prior_eval_interval > 0
                and rel_step % FLAGS.action_prior_eval_interval == 0
            )
            if (
                rel_step % FLAGS.save_interval == 0
                or rel_step in prior_eval_steps
                or periodic_prior_eval
            ):
                save_agent(action_prior, action_prior_save_dir, rel_step)

            if prior_eval_enabled and (
                rel_step in prior_eval_steps or periodic_prior_eval
            ):
                prior_eval_metrics = _evaluate_action_prior(
                    action_prior,
                    env,
                    config_action_prior,
                    prior_num_tasks,
                )
                wandb.log(
                    {**prior_eval_metrics, "rel_step": rel_step},
                    step=i,
                )
                eval_logger.log(prior_eval_metrics, step=i)

        if FLAGS.action_prior_only:
            train_logger.close()
            eval_logger.close()
            env.close()
            return

        action_prior_sample_rng = jax.random.fold_in(
            jax.random.PRNGKey(FLAGS.seed), 0x5052494F
        )
        wandb.log(
            {"config/action_prior_sampling_domain_size": model_train_dataset.size},
            step=pre_model_steps + FLAGS.action_prior_train_steps,
        )

        pre_model_steps += FLAGS.action_prior_train_steps

    # Train "agent"
    _set_dataset_discount(
        (model_train_dataset, model_val_dataset, model_eval_dataset),
        config_model["discount"],
    )
    model_dataset_discount = _get_dataset_discount(model_train_dataset)
    print(
        "[main_gen] starting model stage with dataset discount "
        f"{model_dataset_discount}"
    )
    wandb.log(
        {
            "config/model_dataset_discount": model_dataset_discount,
            "config/agent_discount": config_model["discount"],
        },
        step=pre_model_steps,
    )
    for i in tqdm.tqdm(
        range(
            1 + pre_model_steps,
            FLAGS.train_steps + 1 + pre_model_steps,
        ),
        smoothing=0.1,
        dynamic_ncols=True,
        desc="GHM training",
    ):
        rel_step = i - pre_model_steps
        # Update agent.
        batch = model_train_dataset.sample(config_model["batch_size"])
        if action_prior is not None:
            batch, action_prior_sample_rng = _inject_online_action_prior(
                batch,
                action_prior,
                action_prior_sample_rng,
                use_future_bootstrap=config_model["loss_type"] == "MC_HC_TD2",
                state_mchc=config_model["agent_name"] == "tdflow",
            )
        if config_model["normalize_observations"]:
            batch = dict_normalize(
                batch,
                model_train_dataset.observation_statistics,
                model_train_dataset.norm_method,
            )
        if encode_observations and not model_batches_are_cached_latents:
            batch = dict_encode(batch, pretrained_encoder)
        model, update_info = model.update(batch)
        if "stats" in update_info:
            del update_info["stats"]
        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if model_val_dataset is not None:
                val_batch = model_val_dataset.sample(config_model["batch_size"])
                if action_prior is not None:
                    val_batch, action_prior_sample_rng = _inject_online_action_prior(
                        val_batch,
                        action_prior,
                        action_prior_sample_rng,
                        use_future_bootstrap=(
                            config_model["loss_type"] == "MC_HC_TD2"
                        ),
                        state_mchc=config_model["agent_name"] == "tdflow",
                    )
                if config_model["normalize_observations"]:
                    val_batch = dict_normalize(
                        val_batch,
                        model_val_dataset.observation_statistics,
                        model_val_dataset.norm_method,
                    )
                if encode_observations and not model_batches_are_cached_latents:
                    val_batch = dict_encode(val_batch, pretrained_encoder)
                model, val_loss, val_info = model.compute_loss(batch=val_batch)
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
            # wandb.log(train_metrics, step=i)
            wandb.log({**train_metrics, "rel_step": rel_step}, step=i)
            train_logger.log(train_metrics, step=i)
        eval_logs = None
        if rel_step == 1 or i % FLAGS.eval_interval == 0:
            metric_discount = (
                float(config_model.get("discount", 0.99))
                if FLAGS.metric_discount < 0.0
                else FLAGS.metric_discount
            )
            if hasattr(model, "compute_log_likelihood") and config_model.get(
                "supports_log_likelihood", True
            ):
                # compute validation likelihood
                model, avg_log_likelihood = compute_log_likelihood(
                    model,
                    val_dataset,
                    num_samples_per_source_state=FLAGS.num_eval_samples_per_state,
                    num_eval_states=FLAGS.num_eval_states,
                    discount=metric_discount,
                    config=config_model,
                    pretrained_encoder=(
                        pretrained_encoder if encode_observations else None
                    ),
                    stat_dataset=train_dataset,
                )
                if eval_logs is None:
                    eval_logs = {}
                eval_logs["evaluation/val/log_likelihood"] = avg_log_likelihood.item()
                # compute training likelihood
                model, avg_log_likelihood = compute_log_likelihood(
                    model,
                    train_dataset,
                    num_samples_per_source_state=FLAGS.num_eval_samples_per_state,
                    num_eval_states=FLAGS.num_eval_states,
                    discount=metric_discount,
                    config=config_model,
                    pretrained_encoder=(
                        pretrained_encoder if encode_observations else None
                    ),
                    stat_dataset=train_dataset,
                )
                eval_logs["evaluation/train/log_likelihood"] = avg_log_likelihood.item()
                # comptue eval likelihood if eval dataset is provided
                if eval_dataset is not None:
                    model, avg_log_likelihood = compute_log_likelihood(
                        model,
                        eval_dataset,
                        num_samples_per_source_state=FLAGS.num_eval_samples_per_state,
                        num_eval_states=FLAGS.num_eval_states,
                        discount=metric_discount,
                        config=config_model,
                        pretrained_encoder=(
                            pretrained_encoder if encode_observations else None
                        ),
                        stat_dataset=train_dataset,
                    )
                    eval_logs["evaluation/eval/log_likelihood"] = (
                        avg_log_likelihood.item()
                    )

            if FLAGS.compute_emd_xy:
                if not encode_observations:
                    emd_discount = (
                        metric_discount
                        if FLAGS.emd_discount < 0.0
                        else FLAGS.emd_discount
                    )
                    sample_decoder = None
                    source_encoder = pretrained_encoder if encode_observations else None
                    if val_dataset is not None:
                        model, emd_info = compute_emd_xy(
                            model,
                            val_dataset,
                            config_model,
                            num_source_states=FLAGS.num_emd_trajectories,
                            num_samples_per_source_state=FLAGS.num_emd_samples_per_state,
                            discount=emd_discount,
                            epsilon=FLAGS.emd_epsilon,
                            num_sinkhorn_iters=FLAGS.emd_sinkhorn_iters,
                            source_encoder=source_encoder,
                            sample_decoder=sample_decoder,
                            stat_dataset=train_dataset,
                        )
                        if eval_logs is None:
                            eval_logs = {}
                        eval_logs["evaluation/val/emd_xy"] = emd_info["emd_xy"]
                        eval_logs["evaluation/val/emd_xy_std"] = emd_info["emd_xy_std"]
                    if eval_dataset is not None:
                        model, emd_info = compute_emd_xy(
                            model,
                            eval_dataset,
                            config_model,
                            num_source_states=FLAGS.num_emd_trajectories,
                            num_samples_per_source_state=FLAGS.num_emd_samples_per_state,
                            discount=emd_discount,
                            epsilon=FLAGS.emd_epsilon,
                            num_sinkhorn_iters=FLAGS.emd_sinkhorn_iters,
                            source_encoder=source_encoder,
                            sample_decoder=sample_decoder,
                            stat_dataset=train_dataset,
                        )
                        if eval_logs is None:
                            eval_logs = {}
                        eval_logs["evaluation/eval/emd_xy"] = emd_info["emd_xy"]
                        eval_logs["evaluation/eval/emd_xy_std"] = emd_info["emd_xy_std"]

            if FLAGS.compute_cube_emd:
                if config_model["agent_name"] != "chunk_tdflow":
                    raise ValueError(
                        "--compute_cube_emd currently requires agent=agents/chunk_tdflow.py."
                    )
                if "cube" not in FLAGS.env_name:
                    raise ValueError("--compute_cube_emd requires a cube environment.")
                if eval_logs is None:
                    eval_logs = {}
                cube_emd_datasets = [("val", val_dataset, "random")]
                if eval_dataset is not None:
                    cube_emd_datasets.append(
                        ("eval", eval_dataset, FLAGS.cube_eval_source_mode)
                    )
                for split, cube_dataset, source_mode in cube_emd_datasets:
                    previous_discount = _get_dataset_discount(cube_dataset)
                    _set_dataset_discount((cube_dataset,), metric_discount)
                    try:
                        model, cube_emd = compute_cube_chunk_emd(
                            model,
                            cube_dataset,
                            config_model,
                            num_source_states=FLAGS.num_eval_states,
                            num_samples_per_source_state=FLAGS.num_eval_samples_per_state,
                            epsilon=FLAGS.emd_epsilon,
                            stat_dataset=train_dataset,
                            source_mode=source_mode,
                        )
                    finally:
                        _set_dataset_discount((cube_dataset,), previous_discount)
                    prefix = f"evaluation/{split}"
                    eval_logs[f"{prefix}/emd_cube_raw"] = cube_emd["emd_raw"]
                    eval_logs[f"{prefix}/emd_cube_raw_std"] = cube_emd[
                        "emd_raw_std"
                    ]
                    eval_logs[f"{prefix}/emd_cube_position"] = cube_emd[
                        "emd_cube_position"
                    ]
                    eval_logs[f"{prefix}/emd_cube_position_std"] = cube_emd[
                        "emd_cube_position_std"
                    ]
                    eval_logs[f"{prefix}/cube_empirical_terminal_success_mass"] = (
                        cube_emd["empirical_terminal_success_mass"]
                    )
                    eval_logs[f"{prefix}/cube_generated_terminal_success_mass"] = (
                        cube_emd["generated_terminal_success_mass"]
                    )
                    eval_logs[f"{prefix}/cube_terminal_success_mass_abs_error"] = (
                        cube_emd["terminal_success_mass_abs_error"]
                    )
                    for position, value in enumerate(
                        cube_emd["emd_raw_per_position"], start=1
                    ):
                        eval_logs[
                            f"{prefix}/emd_cube_raw_position_{position}"
                        ] = float(value)
                    for position, value in enumerate(
                        cube_emd["emd_cube_position_per_position"], start=1
                    ):
                        eval_logs[
                            f"{prefix}/emd_cube_position_{position}"
                        ] = float(value)

            if FLAGS.plot_samples:
                if (
                    config_model["agent_name"] == "chunk_tdflow"
                    and "cube" in FLAGS.env_name
                ):
                    previous_val_discount = _get_dataset_discount(val_dataset)
                    _set_dataset_discount(
                        (val_dataset,), metric_discount
                    )
                    try:
                        model, cube_figs = plot_chunk_samples_cube(
                            model,
                            val_dataset,
                            config_model,
                            env.unwrapped,
                            num_samples_per_state=FLAGS.num_eval_samples_per_state,
                            num_plot_states=3,
                            stat_dataset=train_dataset,
                            source_mode="random",
                        )
                    finally:
                        _set_dataset_discount(
                            (val_dataset,), previous_val_discount
                        )
                    figs = {
                        f"cube_chunk_comparison_{mode}": fig
                        for mode, fig in cube_figs.items()
                    }
                    if eval_dataset is not None:
                        previous_eval_discount = _get_dataset_discount(eval_dataset)
                        _set_dataset_discount(
                            (eval_dataset,), metric_discount
                        )
                        try:
                            model, eval_cube_figs = plot_chunk_samples_cube(
                                model,
                                eval_dataset,
                                config_model,
                                env.unwrapped,
                                num_samples_per_state=FLAGS.num_eval_samples_per_state,
                                num_plot_states=3,
                                stat_dataset=train_dataset,
                                source_mode=FLAGS.cube_eval_source_mode,
                            )
                        finally:
                            _set_dataset_discount(
                                (eval_dataset,), previous_eval_discount
                            )
                        for mode, fig in eval_cube_figs.items():
                            wandb.log(
                                {
                                    "evaluation/eval/sample_xy/"
                                    f"cube_chunk_comparison_{mode}": wandb.Image(fig),
                                    "rel_step": rel_step,
                                },
                                step=i,
                            )
                            plt.close(fig)
                elif (
                    config_model["agent_name"] == "chunk_tdflow"
                    and "puzzle-" in FLAGS.env_name
                ):
                    previous_val_discount = _get_dataset_discount(val_dataset)
                    _set_dataset_discount((val_dataset,), metric_discount)
                    try:
                        model, puzzle_figs = plot_chunk_samples_puzzle(
                            model,
                            val_dataset,
                            config_model,
                            env.unwrapped,
                            num_samples_per_state=FLAGS.num_eval_samples_per_state,
                            num_plot_states=3,
                            stat_dataset=train_dataset,
                        )
                    finally:
                        _set_dataset_discount(
                            (val_dataset,), previous_val_discount
                        )
                    figs = {
                        f"puzzle_chunk_comparison_{mode}": fig
                        for mode, fig in puzzle_figs.items()
                    }
                elif "vae" in config_model["agent_name"]:
                    # debug = True if i == 1000 else False
                    debug = False
                    figs = plot_samples_xy_vae(
                        model,
                        val_dataset,
                        config_model,
                        env.unwrapped,
                        num_samples_per_state=FLAGS.num_eval_samples_per_state,
                        num_plot_states=5,
                        plot_images=example_obs.ndim == 4,
                        debug=debug,
                    )
                else:
                    model, figs = plot_samples_xy(
                        model,
                        val_dataset,
                        config_model,
                        env.unwrapped,
                        num_samples_per_state=FLAGS.num_eval_samples_per_state,
                        num_plot_states=5,
                        plot_actions=FLAGS.plot_actions,
                        plot_images="visual" in FLAGS.env_name,
                        plot_maze=("maze" in FLAGS.env_name and (
                            "visual" not in FLAGS.env_name
                        )),
                        pretrained_encoder=(
                            pretrained_encoder if encode_observations else None
                        ),
                        pretrained_decoder=pretrained_decoder,
                        stat_dataset=train_dataset,
                    )

                for name, fig in figs.items():
                    if fig is None:
                        continue
                    wandb.log(
                        {
                            f"evaluation/val/sample_xy/{name}": wandb.Image(fig),
                            "rel_step": rel_step,
                        },
                        step=i,
                    )

            plt.close("all")
            if eval_logs is not None:
                wandb.log({**eval_logs, "rel_step": rel_step}, step=i)
                eval_logger.log(eval_logs, step=i)

        if i % FLAGS.save_interval == 0:
            save_agent(model, FLAGS.save_dir, i)

    # transfer patch encoder to policy


    policy_restored_steps = 0
    policy_global_offset = FLAGS.train_steps + pre_model_steps
    if policy_restore_path is not None:
        if policy_model is None or config_policy["agent_name"] != "gcfbc":
            raise ValueError("Policy restore currently requires a GCFBC policy.")
        initial_optimizer_step = int(policy_model.network.step)
        policy_model = restore_agent(
            policy_model, policy_restore_path, FLAGS.policy_restore_epoch
        )
        policy_restored_steps = int(policy_model.network.step) - initial_optimizer_step
        if policy_restored_steps < 0:
            raise ValueError("Restored policy has an invalid optimizer step.")
        policy_global_offset = max(policy_global_offset, FLAGS.policy_restore_epoch)
        print(f"[policy restore] retained optimizer state at {policy_restored_steps} updates; "
              f"training {FLAGS.policy_train_steps} additional updates")
    # Now train policy if specified
    if policy_cls is not None:
        _set_dataset_loss_type(
            (policy_train_dataset, policy_val_dataset, policy_eval_dataset), None
        )
        _set_dataset_discount(
            (policy_train_dataset, policy_val_dataset, policy_eval_dataset),
            config_policy["discount"],
        )
        policy_dataset_discount = _get_dataset_discount(policy_train_dataset)
        print(
            "[main_gen] starting policy stage with dataset discount "
            f"{policy_dataset_discount}"
        )
        wandb.log(
            {
                "config/policy_dataset_discount": policy_dataset_discount,
                "config/policy_discount": config_policy["discount"],
            },
            step=FLAGS.train_steps + pre_model_steps,
        )
        n_augment = int(config_policy["batch_size"] * FLAGS.gen_prop)
        wandb.log(
            {
                "config/n_augment": n_augment,
            },
            step=FLAGS.train_steps + pre_model_steps,
        )
        rollout_goal_cache = None
        if FLAGS.augmentation_mode == "onestep_ram_cache":
            from utils.rollout_goal_cache import build_rollout_goal_cache

            if (
                encode_observations
                or pretrained_encoder is not None
                or generated_goal_transform is not None
            ):
                raise ValueError("RAM caching does not yet support encoders or goal transforms")
            if (policy_train_dataset.config.get("frame_stack") is not None
                    or policy_train_dataset.config.get("p_aug", 0)
                    or policy_train_dataset.config.get("observation_history_size", 1) != 1):
                raise ValueError("RAM caching requires unchanged vector policy observations")
            if n_augment <= 0:
                raise ValueError("RAM caching requires at least one augmented example per batch")
            source_data = policy_train_dataset.dataset
            cache_config = dict(model.config)
            cache_bytes = (source_data.size * int(cache_config["sample_max_steps"])
                           * source_data["observations"].shape[-1] * 4)
            print(f"[rollout cache] allocating {cache_bytes / 2**30:.3f} GiB in CPU RAM", flush=True)
            cache_started = time.time()
            last_cache_report = [0.0]

            def cache_progress(done, total, elapsed):
                if elapsed - last_cache_report[0] >= 30 or done == total:
                    print(f"[rollout cache] {done}/{total} rows, {elapsed:.1f}s", flush=True)
                    last_cache_report[0] = elapsed

            rollout_goal_cache, model = build_rollout_goal_cache(
                model, source_data, cache_config, model_train_dataset,
                batch_size=FLAGS.rollout_cache_batch_size, seed=FLAGS.seed,
                progress=cache_progress,
            )
            wandb.log({"config/rollout_cache_bytes": cache_bytes,
                       "config/rollout_cache_steps": int(cache_config["sample_max_steps"]),
                       "config/rollout_cache_discount": float(cache_config["discount"]),
                       "time/rollout_cache_build_seconds": time.time() - cache_started},
                      step=FLAGS.train_steps + pre_model_steps)
            # Policy evaluation and saving use policy_model, not the generator.
            model = None
        policy_stage_start = policy_global_offset + 1
        for i in tqdm.tqdm(
            range(
                policy_stage_start,
                policy_stage_start + FLAGS.policy_train_steps,
            ),
            smoothing=0.1,
            dynamic_ncols=True,
            desc="Policy training",
        ):
            rel_step = (
                policy_restored_steps + i - policy_global_offset
            )
            # Update agent.
            if rollout_goal_cache is None:
                batch = policy_train_dataset.sample(config_policy["batch_size"])
            else:
                policy_indices = policy_train_dataset.get_random_idxs(config_policy["batch_size"])
                batch = policy_train_dataset.sample(config_policy["batch_size"], idxs=policy_indices)

            if encode_observations and not policy_batches_are_cached_latents:
                batch = dict_encode(batch, pretrained_encoder)
            if rollout_goal_cache is not None:
                batch = rollout_goal_cache.augment(batch, policy_indices, n_augment)
            elif n_augment > 0:
                batch, model = gen_augment(
                    batch,
                    policy_train_dataset,
                    n_augment,
                    model,
                    config_model,
                    generated_goal_transform=generated_goal_transform,
                    gen_stat_dataset=model_train_dataset,
                )


            policy_model, update_info = policy_model.update(batch)
            if "stats" in update_info:
                del update_info["stats"]
            for key in (
                "aug_real_goal_norm",
                "aug_model_goal_norm",
                "aug_goal_norm_ratio",
            ):
                if key in batch:
                    update_info[key] = batch[key]

            # Log metrics.
            if i % FLAGS.log_interval == 0 or rel_step == 1:
                train_metrics = {
                    f"policy_training/{k}": v for k, v in update_info.items()
                }
                if policy_val_dataset is not None:
                    val_batch = policy_val_dataset.sample(
                        config_policy["batch_size"]
                    )  # note, no aug on val
                    if encode_observations and not policy_batches_are_cached_latents:
                        val_batch = dict_encode(val_batch, pretrained_encoder)
                    _, val_info = policy_model.total_loss(val_batch, grad_params=None)
                    if "stats" in val_info:
                        del val_info["stats"]
                    train_metrics.update(
                        {f"policy_validation/{k}": v for k, v in val_info.items()}
                    )
                if policy_eval_dataset is not None:
                    eval_batch = policy_eval_dataset.sample(
                        config_policy["batch_size"]
                    )  # note, no aug on eval
                    if encode_observations and not policy_batches_are_cached_latents:
                        eval_batch = dict_encode(eval_batch, pretrained_encoder)
                    _, eval_info = policy_model.total_loss(eval_batch, grad_params=None)
                    if "stats" in eval_info:
                        del eval_info["stats"]
                    train_metrics.update(
                        {f"policy_evaluation/{k}": v for k, v in eval_info.items()}
                    )
                train_metrics["time/epoch_time"] = (
                    time.time() - last_time
                ) / FLAGS.log_interval
                train_metrics["time/total_time"] = time.time() - first_time
                last_time = time.time()
                # wandb.log(train_metrics, step=i)
                wandb.log({**train_metrics, "rel_step": rel_step}, step=i)
                train_logger.log(train_metrics, step=i)

            if (
                (i % FLAGS.policy_eval_interval == 0 or rel_step == 1)
                and FLAGS.eval_episodes > 0
            ):
                if FLAGS.eval_on_cpu:
                    eval_agent = jax.device_put(
                        policy_model, device=jax.devices("cpu")[0]
                    )
                else:
                    eval_agent = policy_model
                renders = []
                eval_metrics = {}
                overall_metrics = defaultdict(list)
                task_infos = (
                    env.unwrapped.task_infos
                    if hasattr(env.unwrapped, "task_infos")
                    else env.task_infos
                )
                num_tasks = (
                    FLAGS.eval_tasks
                    if FLAGS.eval_tasks is not None
                    else len(task_infos)
                )
                for task_id in tqdm.trange(1, num_tasks + 1):
                    task_name = task_infos[task_id - 1]["task_name"]
                    eval_info, _, cur_renders = evaluate(
                        agent=eval_agent,
                        env=env,
                        task_id=task_id,
                        config=config_policy,
                        num_eval_episodes=FLAGS.eval_episodes,
                        num_video_episodes=FLAGS.video_episodes,
                        video_frame_skip=FLAGS.video_frame_skip,
                        eval_temperature=FLAGS.eval_temperature,
                        eval_gaussian=FLAGS.eval_gaussian,
                    )
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
                if "evaluation/overall_success" in eval_metrics:
                    successes.append(eval_metrics["evaluation/overall_success"])
                if FLAGS.video_episodes > 0:
                    video = get_wandb_video(renders=renders, n_cols=num_tasks)
                    eval_metrics["video"] = video

                # wandb.log(eval_metrics, step=i)
                wandb.log({**eval_metrics, "rel_step": rel_step}, step=i)
                eval_logger.log(eval_metrics, step=i)

            if i % FLAGS.save_interval == 0:
                save_agent(policy_model, FLAGS.save_dir, i)

    # add to wandb summary
    train_logger.close()
    eval_logger.close()
    renderer = getattr(env.unwrapped, "_renderer", None)
    if renderer is not None:
        renderer.close()
        env.unwrapped._renderer = None
    env.close()


if __name__ == "__main__":
    app.run(main)
