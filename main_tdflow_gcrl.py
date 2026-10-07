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
import gc
import json
import random
import time
from collections import defaultdict

import jax
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.run_restore import load_run_flags as _load_run_flags, resolve_run_directory as _resolve_run_directory
from ogbench.env_loader import make_dataset, make_env_and_dataset
from utils.datasets import (
    AugGCDataset,
    Dataset,
    GCDataset,
    HGCDataset,
    dict_normalize,
)
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.gen_evaluation import compute_log_likelihood, plot_samples_xy
from utils.log_utils import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    get_wandb_video,
    setup_wandb,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "antmaze-large-v0", "Environment name.")
flags.DEFINE_string("dataset_path", None, "Dataset path.")
flags.DEFINE_integer("same_eval", 0, "Whether to use the same dataset for evaluation.")
flags.DEFINE_string("eval_dataset_path", None, "Evaluation dataset path.")
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("wandb_mode", "online", "wandb mode")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")
flags.DEFINE_string(
    "gen_ae_path",
    None,
    "Path to AE checkpoint run used only for augmented-goal generation.",
)
flags.DEFINE_integer("gen_ae_epoch", None, "Checkpoint epoch for gen_ae_path.")

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
flags.DEFINE_integer(
    "plot_actions", 0, "Whether to plot actions in the evaluation graphs."
)
flags.DEFINE_integer("video_episodes", 1, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")
flags.DEFINE_integer("eval_on_cpu", 0, "Whether to evaluate on CPU.")

flags.DEFINE_string("project", "comb-gen-gen", "WandB project name.")

config_flags.DEFINE_config_file("agent", "agents/tdflow.py", lock_config=False)
config_flags.DEFINE_config_file("policy", None, lock_config=False)


def _restore_ae_from_run(run_dir, epoch, example_obs, example_actions):
    if epoch is None:
        raise ValueError("gen_ae_epoch must be set when providing gen_ae_path.")
    flag_dict = _load_run_flags(run_dir)
    config = flag_dict.get("encoder", flag_dict.get("agent"))
    if config is None:
        raise ValueError(
            f"flags.json in {run_dir} has neither encoder nor agent config."
        )
    agent_name = config.get("agent_name")
    if agent_name not in agents:
        raise ValueError(f"Unknown AE agent {agent_name} in {run_dir}")
    ae = agents[agent_name].create(
        flag_dict.get("seed", 0),
        example_obs,
        example_actions,
        config,
    )
    ae = restore_agent(ae, run_dir, epoch)
    if not hasattr(ae, "encode") or not hasattr(ae, "decode"):
        raise ValueError(
            "gen_ae_path must restore an object with encode() and decode()."
        )
    return ae, config


def main(_):
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
    if config_model.get("n_step", None):
        raise ValueError("Legacy n_step frame stacking is not supported; use observation history.")

    env, train_dataset, val_dataset = make_env_and_dataset(
        FLAGS.env_name,
        FLAGS.dataset_path,
        frame_stack=config_model["frame_stack"],
        precomputed_features=config_model["precomputed_features"],
        implemtented=True,
        same_eval=bool(FLAGS.same_eval),
    )
    eval_dataset = None
    if FLAGS.eval_dataset_path is not None:
        eval_dataset = make_dataset(
            FLAGS.env_name,
            FLAGS.eval_dataset_path,
            precomputed_features=config_model["precomputed_features"],
            flatten_obs=False,
        )


    # if we're using augmented goals in the dataset class, we handle this now.
    model_dataset_class = {
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
    }[config_model["dataset_class"]]
    model_train_dataset = model_dataset_class(
        Dataset.create(**train_dataset), config_model
    )
    if val_dataset is not None:
        model_val_dataset = model_dataset_class(
            Dataset.create(**val_dataset), config_model
        )
    model_eval_dataset = None
    if eval_dataset is not None:
        model_eval_dataset = model_dataset_class(
            Dataset.create(**eval_dataset), config_model
        )

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = model_train_dataset.sample(1)
    if config_model["discrete"]:
        raise ValueError("Only continuous actions are supported.")

    example_obs = example_batch["observations"]

    gen_ae = None
    model_example_obs = example_obs
    if FLAGS.gen_ae_path is not None:
        if FLAGS.restore_path is None:
            raise ValueError(
                "gen_ae_path requires restore_path; the AE-latent path is for "
                "reloading a latent flow for augmented goal generation."
            )
        if FLAGS.train_steps != 0:
            raise ValueError(
                "gen_ae_path currently supports restored-flow policy training only; "
                "set train_steps=0 so the flow is used only for augmented goal generation."
            )
        gen_ae_run = _resolve_run_directory(FLAGS.gen_ae_path)
        gen_ae, gen_ae_config = _restore_ae_from_run(
            gen_ae_run,
            FLAGS.gen_ae_epoch,
            example_obs,
            example_batch["actions"],
        )
        model_example_obs = gen_ae.encode(example_obs)
        print(
            f"Restored generation AE from {gen_ae_run} "
            f"at epoch {FLAGS.gen_ae_epoch}."
        )
        config_model["gen_ae_config"] = gen_ae_config

    model_cls = agents[config_model["agent_name"]]
    model = model_cls.create(
        FLAGS.seed,
        model_example_obs,
        example_batch["actions"],
        config_model,
    )

    # Restore model.
    if FLAGS.restore_path is not None:
        model = restore_agent(model, FLAGS.restore_path, FLAGS.restore_epoch)
        print(
            f'Restored agent from {os.path.join(FLAGS.restore_path, f"params_{FLAGS.restore_epoch}.pkl")}'
        )

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()
    successes = []
    for i in tqdm.tqdm(
        range(1, FLAGS.train_steps + 1),
        desc="td flow training",
        smoothing=0.1,
        dynamic_ncols=True,
    ):
        # Update agent.
        batch = model_train_dataset.sample(config_model["batch_size"])
        if config_model["normalize_observations"]:
            batch = dict_normalize(
                batch,
                model_train_dataset.observation_statistics,
                model_train_dataset.norm_method,
            )
        model, update_info = model.update(batch)
        if "stats" in update_info:
            del update_info["stats"]
        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = model_val_dataset.sample(config_model["batch_size"])
                if config_model["normalize_observations"]:
                    val_batch = dict_normalize(
                        val_batch,
                        model_val_dataset.observation_statistics,
                        model_val_dataset.norm_method,
                    )
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
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i == 1 or i % FLAGS.eval_interval == 0:
            # compute validation likelihood
            model, avg_log_likelihood = compute_log_likelihood(
                model,
                model_val_dataset,
                num_samples_per_source_state=FLAGS.num_eval_samples_per_state,
                num_eval_states=FLAGS.num_eval_states,
                config=config_model,
                stat_dataset=model_train_dataset,
            )
            eval_logs = {}
            eval_logs["evaluation/val/log_likelihood"] = avg_log_likelihood.item()

            # compute training likelihood
            model, avg_log_likelihood = compute_log_likelihood(
                model,
                model_train_dataset,
                num_samples_per_source_state=FLAGS.num_eval_samples_per_state,
                num_eval_states=FLAGS.num_eval_states,
                config=config_model,
                stat_dataset=model_train_dataset,
            )
            eval_logs["evaluation/train/log_likelihood"] = avg_log_likelihood.item()
            if model_eval_dataset is not None:
                model, avg_log_likelihood = compute_log_likelihood(
                    model,
                    model_eval_dataset,
                    num_samples_per_source_state=FLAGS.num_eval_samples_per_state,
                    num_eval_states=FLAGS.num_eval_states,
                    config=config_model,
                    stat_dataset=model_train_dataset,
                )
                eval_logs["evaluation/eval/log_likelihood"] = avg_log_likelihood.item()

            if "maze" in FLAGS.env_name:
                model, figs = plot_samples_xy(
                    model,
                    model_val_dataset,
                    config_model,
                    env.unwrapped,
                    num_samples_per_state=FLAGS.num_eval_samples_per_state,
                    num_plot_states=5,
                    plot_actions=FLAGS.plot_actions,
                    plot_images="visual" in FLAGS.env_name,
                    plot_maze="visual" not in FLAGS.env_name,
                    stat_dataset=model_train_dataset,
                )
                for name, fig in figs.items():
                    if fig is None:
                        continue
                    wandb.log(
                        {f"evaluation/val/sample_xy/{name}": wandb.Image(fig)}, step=i
                    )
                    if hasattr(fig, "canvas"):
                        plt.close(fig)

            wandb.log(eval_logs, step=i)
            eval_logger.log(eval_logs, step=i)

        if i % FLAGS.save_interval == 0:
            save_agent(model, FLAGS.save_dir, i)

    # Flow training can leave large compiled executables cached. Release what we
    # can before initializing the image CRL policy and its separate encoder stack.
    jax.clear_caches()
    gc.collect()

    # Before training the policy, we initialize both it and the policy dataset.
    policy_dataset_class = {
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
        "AugGCDataset": AugGCDataset,
    }[config_policy["dataset_class"]]
    policy_eval_dataset = None
    if config_policy["dataset_class"] in ["GCDataset", "HGCDataset"]:
        policy_train_dataset = policy_dataset_class(
            Dataset.create(**train_dataset), config_policy
        )
        if val_dataset is not None:
            policy_val_dataset = policy_dataset_class(
                Dataset.create(**val_dataset), config_policy
            )
        if eval_dataset is not None:
            policy_eval_dataset = policy_dataset_class(
                Dataset.create(**eval_dataset), config_policy
            )
    else:
        policy_train_dataset = policy_dataset_class(
            Dataset.create(**train_dataset),
            config_policy,
            config_model,
            model,
            gen_ae=gen_ae,
        )
        if val_dataset is not None:
            policy_val_dataset = policy_dataset_class(
                Dataset.create(**val_dataset),
                config_policy,
                config_model,
                model,
                gen_ae=gen_ae,
            )
        if eval_dataset is not None:
            policy_eval_dataset = policy_dataset_class(
                Dataset.create(**eval_dataset),
                config_policy,
                config_model,
                model,
                gen_ae=gen_ae,
            )

    policy_cls = None
    if isinstance(policy_train_dataset, AugGCDataset):
        example_policy_batch = policy_train_dataset.sample(1, aug_goals=False)
    else:
        example_policy_batch = policy_train_dataset.sample(1)
    if config_policy is not None:
        policy_cls = agents[config_policy["agent_name"]]
        policy_model = policy_cls.create(
            FLAGS.seed,
            example_obs,
            example_policy_batch["actions"],
            config_policy,
        )

    # Now train policy if specified
    if policy_cls is not None:
        for i in tqdm.tqdm(
            range(
                FLAGS.train_steps + 1, FLAGS.train_steps + FLAGS.policy_train_steps + 1
            ),
            desc="policy training",
            smoothing=0.1,
            dynamic_ncols=True,
        ):
            # Update agent.
            batch = policy_train_dataset.sample(config_policy["batch_size"])

            policy_model, update_info = policy_model.update(batch)
            if "stats" in update_info:
                del update_info["stats"]

            # Log metrics.
            if i % FLAGS.log_interval == 0 or i == FLAGS.train_steps + 1:
                train_metrics = {
                    f"policy_training/{k}": v for k, v in update_info.items()
                }
                if val_dataset is not None:
                    # val_batch = policy_val_dataset.sample(config_policy["batch_size"])
                    val_batch = policy_val_dataset.sample(
                        config_policy["batch_size"], aug_goals=False
                    )
                    _, val_info = policy_model.total_loss(val_batch, grad_params=None)
                    if "stats" in val_info:
                        del val_info["stats"]
                    train_metrics.update(
                        {f"policy_validation/{k}": v for k, v in val_info.items()}
                    )
                if eval_dataset is not None:
                    # eval_batch = policy_eval_dataset.sample(config_policy["batch_size"])
                    eval_batch = policy_eval_dataset.sample(
                        config_policy["batch_size"], aug_goals=False
                    )
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
                wandb.log(train_metrics, step=i)
                train_logger.log(train_metrics, step=i)

            if i % FLAGS.policy_eval_interval == 0 or i == FLAGS.train_steps + 1:
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
                    eval_info, trajs, cur_renders = evaluate(
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

                wandb.log(eval_metrics, step=i)
                eval_logger.log(eval_metrics, step=i)

            if i % FLAGS.save_interval == 0:
                save_agent(policy_model, FLAGS.save_dir, i)

    # add to wandb summary
    train_logger.close()
    eval_logger.close()


if __name__ == "__main__":
    app.run(main)
