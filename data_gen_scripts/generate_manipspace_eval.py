from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from tqdm import tqdm

import ogbench.manipspace  # noqa: F401
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "cube-double-v0", "Cube environment name.")
flags.DEFINE_string("save_path", None, "Output NPZ path.")
flags.DEFINE_list(
    "task_ids",
    None,
    "One-based task IDs to collect (default: all tasks).",
)
flags.DEFINE_integer(
    "demos_per_task",
    1,
    "Number of successful demonstrations to collect for each task.",
)
flags.DEFINE_float("noise", 0.1, "Plan-oracle action noise level.")
flags.DEFINE_float(
    "noise_smoothing",
    0.5,
    "Temporal smoothing applied to plan-oracle action noise.",
)
flags.DEFINE_integer(
    "max_episode_steps",
    None,
    "Episode limit (default: the registered environment limit).",
)
flags.DEFINE_integer(
    "max_attempts_per_demo",
    100,
    "Maximum attempts for each successful task demonstration.",
)


def _target_info(env, info, block_idx, target_pos):
    """Add the target fields expected by CubePlanOracle in task mode."""
    oracle_info = dict(info)
    oracle_info["privileged/target_task"] = "cube"
    oracle_info["privileged/target_block"] = block_idx
    oracle_info["privileged/target_block_pos"] = np.asarray(target_pos).copy()
    oracle_info["privileged/target_block_yaw"] = np.zeros(1, dtype=np.float32)
    return oracle_info


def _find_staging_position(positions, goals):
    candidates = np.array(
        [
            [0.325, -0.25, 0.02],
            [0.525, -0.25, 0.02],
            [0.325, 0.25, 0.02],
            [0.525, 0.25, 0.02],
            [0.425, 0.0, 0.02],
        ]
    )
    occupied = np.concatenate([positions, goals], axis=0)
    clearance = np.min(
        np.linalg.norm(candidates[:, None, :2] - occupied[None, :, :2], axis=-1),
        axis=1,
    )
    return candidates[np.argmax(clearance)].copy()


def _make_move_sequence(positions, goals):
    """Plan cube moves, inserting a staging move for permutation cycles."""
    positions = np.asarray(positions).copy()
    goals = np.asarray(goals)
    pending = {
        idx
        for idx in range(len(goals))
        if np.linalg.norm(positions[idx] - goals[idx]) > 0.03
    }
    sequence = []

    while pending:
        available = []
        for idx in pending:
            blockers = [
                other
                for other in range(len(positions))
                if other != idx
                and np.linalg.norm(positions[other] - goals[idx]) < 0.03
            ]
            if not blockers:
                available.append(idx)

        if available:
            # Lower targets must be placed before cubes stacked above them.
            idx = min(available, key=lambda block_idx: (goals[block_idx, 2], block_idx))
            sequence.append((idx, goals[idx].copy(), "goal"))
            positions[idx] = goals[idx]
            pending.remove(idx)
        else:
            # A swap cycle has no free destination. Move one cube aside first.
            idx = min(pending)
            staging_position = _find_staging_position(positions, goals)
            sequence.append((idx, staging_position, "staging"))
            positions[idx] = staging_position

    return sequence


def _collect_attempt(
    env,
    task_id,
    oracle,
    reset_seed,
    complete_oracle_plan=False,
    oracle_settle_steps=0,
):
    observation, info = env.reset(seed=reset_seed, options=dict(task_id=task_id))
    raw_env = env.unwrapped
    positions = np.array(
        [
            raw_env._data.joint(f"object_joint_{idx}").qpos[:3].copy()
            for idx in range(raw_env._num_cubes)
        ]
    )
    goals = np.array(
        [
            raw_env._data.mocap_pos[raw_env._cube_target_mocap_ids[idx]].copy()
            for idx in range(raw_env._num_cubes)
        ]
    )
    move_sequence = _make_move_sequence(positions, goals)
    episode = defaultdict(list)
    done = False
    completed_moves = 0

    for block_idx, target_pos, move_kind in move_sequence:
        oracle_info = _target_info(env, info, block_idx, target_pos)
        oracle.reset(observation, oracle_info)

        while not done:
            oracle_info = _target_info(env, info, block_idx, target_pos)
            action = np.clip(
                np.asarray(oracle.select_action(observation, oracle_info)), -1, 1
            )
            next_observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode["observations"].append(observation)
            episode["actions"].append(action)
            episode["terminals"].append(done)
            episode["qpos"].append(info["prev_qpos"])
            episode["qvel"].append(info["prev_qvel"])
            episode["success"].append(info["success"])
            episode["terminated"].append(terminated)
            episode["truncated"].append(truncated)
            episode["task_ids"].append(task_id)
            episode["move_is_staging"].append(move_kind == "staging")

            observation = next_observation
            if done or oracle.done:
                break

        if complete_oracle_plan and oracle.done and not done:
            for _ in range(oracle_settle_steps):
                oracle_info = _target_info(env, info, block_idx, target_pos)
                action = np.clip(
                    np.asarray(oracle.select_action(observation, oracle_info)), -1, 1
                )
                next_observation, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode["observations"].append(observation)
                episode["actions"].append(action)
                episode["terminals"].append(done)
                episode["qpos"].append(info["prev_qpos"])
                episode["qvel"].append(info["prev_qvel"])
                episode["success"].append(info["success"])
                episode["terminated"].append(terminated)
                episode["truncated"].append(truncated)
                episode["task_ids"].append(task_id)
                episode["move_is_staging"].append(move_kind == "staging")

                observation = next_observation
                if done:
                    break

        if oracle.done:
            completed_moves += 1
        if done:
            break

    success = bool(info.get("success", False))
    if complete_oracle_plan:
        success = success and completed_moves == len(move_sequence)
        if success and episode["terminals"]:
            # Non-terminating task mode lets the oracle release and retract.
            # Add the offline trajectory boundary after the full plan completes.
            episode["terminals"][-1] = True
    return episode, success


def _as_array(key, values):
    if key in {
        "terminals",
        "success",
        "terminated",
        "truncated",
        "move_is_staging",
    }:
        dtype = bool
    elif key == "task_ids":
        dtype = np.int32
    else:
        dtype = np.float32
    return np.asarray(values, dtype=dtype)


def main(_):
    if FLAGS.save_path is None:
        raise ValueError("--save_path is required.")
    if "cube" not in FLAGS.env_name:
        raise ValueError("This collector currently supports Cube environments only.")
    if FLAGS.demos_per_task <= 0:
        raise ValueError("--demos_per_task must be positive.")

    np.random.seed(FLAGS.seed)
    env_kwargs = dict(terminate_at_goal=True, mode="task")
    if FLAGS.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = FLAGS.max_episode_steps
    env = gymnasium.make(FLAGS.env_name, **env_kwargs)

    if FLAGS.task_ids is None:
        task_ids = list(range(1, env.unwrapped.num_tasks + 1))
    else:
        task_ids = [int(task_id) for task_id in FLAGS.task_ids]
    if not task_ids or any(
        task_id < 1 or task_id > env.unwrapped.num_tasks for task_id in task_ids
    ):
        raise ValueError(
            f"--task_ids must be within [1, {env.unwrapped.num_tasks}]."
        )

    oracle = CubePlanOracle(
        env=env,
        noise=FLAGS.noise,
        noise_smoothing=FLAGS.noise_smoothing,
    )
    dataset = defaultdict(list)
    lengths = defaultdict(list)
    attempts = defaultdict(int)

    progress = tqdm(total=FLAGS.demos_per_task * len(task_ids))
    for demo_idx in range(FLAGS.demos_per_task):
        for task_id in task_ids:
            for _ in range(FLAGS.max_attempts_per_demo):
                attempts[task_id] += 1
                reset_seed = FLAGS.seed + 1_000_000 * demo_idx + attempts[task_id]
                episode, success = _collect_attempt(
                    env, task_id, oracle, reset_seed=reset_seed
                )
                if not success:
                    continue

                for key, values in episode.items():
                    dataset[key].extend(values)
                lengths[task_id].append(len(episode["actions"]))
                progress.update(1)
                progress.set_postfix(task=task_id, length=lengths[task_id][-1])
                break
            else:
                raise RuntimeError(
                    f"Could not collect task {task_id}, demo {demo_idx} after "
                    f"{FLAGS.max_attempts_per_demo} attempts."
                )
    progress.close()
    env.close()

    output = {key: _as_array(key, values) for key, values in dataset.items()}
    np.savez_compressed(FLAGS.save_path, **output)
    print(f"Saved {sum(len(value) for value in lengths.values())} demonstrations to {FLAGS.save_path}")
    print(f"Total transitions: {len(output['observations'])}")
    print(f"Per-task lengths: {dict(lengths)}")
    print(f"Per-task attempts: {dict(attempts)}")


if __name__ == "__main__":
    app.run(main)
