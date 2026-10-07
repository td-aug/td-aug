import glob
import json
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from tqdm import tqdm

from agents import SACAgent
from utils.evaluation import supply_rng
from utils.flax_utils import restore_agent

import ogbench.locomaze  # noqa

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "antmaze-large-v0", "Environment name.")
flags.DEFINE_string("restore_path", "experts/ant", "Expert agent restore path.")
flags.DEFINE_integer("restore_epoch", 400000, "Expert agent restore epoch.")
flags.DEFINE_string("save_path", None, "Save path.")

flags.DEFINE_integer(
    "num_timesteps",
    100_000,
    "Minimum number of training timesteps to collect.",
)
flags.DEFINE_integer(
    "num_val_timesteps",
    0,
    "Minimum number of validation timesteps to collect (0 disables validation).",
)

flags.DEFINE_integer(
    "eval_tasks",
    None,
    "Number of evaluation tasks to use (None for all).",
)
flags.DEFINE_float("noise", 0.2, "Gaussian action noise level.")

flags.DEFINE_integer(
    "max_episode_steps",
    None,
    "Override max episode steps (None uses env default).",
)
flags.DEFINE_boolean(
    "terminate",
    False,
    "Terminate episodes when the goal is reached.",
)
flags.DEFINE_integer(
    "min_episode_steps",
    0,
    "Minimum number of steps in an episode, will not accept if less.",
)


def _get_task_infos(env):
    task_infos = (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, "task_infos")
        else env.task_infos
    )
    num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
    if num_tasks > len(task_infos):
        raise ValueError(
            f"eval_tasks={num_tasks} exceeds available tasks={len(task_infos)}"
        )
    return task_infos[:num_tasks]


def main(_):
    if FLAGS.save_path is None:
        raise ValueError("save_path must be set.")

    np.random.seed(FLAGS.seed)

    env_kwargs = dict(terminate_at_goal=FLAGS.terminate)
    if FLAGS.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = FLAGS.max_episode_steps
    env = gymnasium.make(FLAGS.env_name, **env_kwargs)

    ob_dim = env.observation_space.shape[0]

    # Initialize oracle agent (same as generate_locomaze.py).
    if "point" in FLAGS.env_name:

        def actor_fn(ob, temperature):
            return ob[-2:]

    else:
        restore_path = FLAGS.restore_path
        candidates = glob.glob(restore_path)
        assert len(candidates) == 1, f"Found {len(candidates)} candidates: {candidates}"

        with open(candidates[0] + "/flags.json", "r") as f:
            agent_config = json.load(f)["agent"]

        agent = SACAgent.create(
            FLAGS.seed,
            np.zeros(ob_dim),
            env.action_space.sample(),
            agent_config,
        )
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        actor_fn = supply_rng(agent.sample_actions, rng=agent.rng)

    task_infos = _get_task_infos(env)
    num_tasks = len(task_infos)
    task_ids = list(range(1, num_tasks + 1))

    min_num_train_steps = FLAGS.num_timesteps
    min_num_val_steps = max(0, FLAGS.num_val_timesteps or 0)
    final_total_steps = min_num_train_steps + min_num_val_steps

    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0

    attempt_counts = {task_id: 0 for task_id in task_ids}
    success_counts = {task_id: 0 for task_id in task_ids}
    task_lengths = {task_id: [] for task_id in task_ids}

    task_index = 0
    with tqdm(total=final_total_steps) as pbar:
        while True:
            if (
                total_train_steps >= min_num_train_steps
                and total_steps - total_train_steps >= min_num_val_steps
            ):
                if task_index == 0:
                    break

            task_id = task_ids[task_index]
            attempt_counts[task_id] += 1

            ob, _ = env.reset(options=dict(task_id=task_id))

            done = False
            step = 0
            episode = defaultdict(list)
            episode_success = False

            while not done:
                # Get the oracle subgoal and compute the direction.
                subgoal_xy, _ = env.unwrapped.get_oracle_subgoal(
                    env.unwrapped.get_xy(), env.unwrapped.cur_goal_xy
                )
                subgoal_dir = subgoal_xy - env.unwrapped.get_xy()
                subgoal_dir = subgoal_dir / (np.linalg.norm(subgoal_dir) + 1e-6)

                agent_ob = env.unwrapped.get_ob(ob_type="states")
                # Exclude the agent's position and add the subgoal direction.
                agent_ob = np.concatenate([agent_ob[2:], subgoal_dir])
                action = actor_fn(agent_ob, temperature=0)
                # Add Gaussian noise to the action.
                action = action + np.random.normal(0, FLAGS.noise, action.shape)
                action = np.clip(action, -1, 1)

                next_ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode["observations"].append(ob)
                episode["actions"].append(action)
                episode["terminals"].append(done)
                if "prev_qpos" in info:
                    episode["qpos"].append(info["prev_qpos"])
                if "prev_qvel" in info:
                    episode["qvel"].append(info["prev_qvel"])
                if "success" in info:
                    episode["success"].append(info["success"])
                    if info["success"] == 1.0:
                        episode_success = True

                ob = next_ob
                step += 1

            if not episode_success or step < FLAGS.min_episode_steps:
                continue

            success_counts[task_id] += 1
            task_lengths[task_id].append(step)
            task_index = (task_index + 1) % num_tasks

            for k, v in episode.items():
                dataset[k].extend(v)

            if total_train_steps < min_num_train_steps:
                total_train_steps += step
            total_steps += step
            pbar.update(step)

    print("Total steps:", total_steps)
    print("Total successful episodes:", sum(success_counts.values()))
    print("Per-task successes:", success_counts)
    print("Per-task attempts:", attempt_counts)
    avg_len_per_task = {
        task_id: (np.mean(lengths) if len(lengths) > 0 else 0.0)
        for task_id, lengths in task_lengths.items()
    }
    print("Average length per task:", avg_len_per_task)
    
    train_dataset = {}
    for k, v in dataset.items():
        if "observations" in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k in {"terminals", "success"}:
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)

    np.savez_compressed(FLAGS.save_path, **train_dataset)

    if min_num_val_steps > 0:
        val_path = FLAGS.save_path.replace(".npz", "-val.npz")
        val_dataset = {
            k: np.array(v[total_train_steps:], dtype=train_dataset[k].dtype)
            for k, v in dataset.items()
        }
        np.savez_compressed(val_path, **val_dataset)


if __name__ == "__main__":
    app.run(main)
