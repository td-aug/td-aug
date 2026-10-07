from collections import defaultdict

import jax
import numpy as np
from tqdm import trange

from utils.goal_representations import (
    cube_oracle_representation,
    locomaze_oracle_representation,
    puzzle_oracle_representation,
    scene_oracle_representation,
)


def _oracle_goal_representation(goal, env):
    """Project an evaluation goal to the environment's oracle representation."""
    if goal is None:
        raise ValueError("Oracle goal representation requires an environment goal.")
    env_id = env.spec.id.lower()
    if "antmaze" in env_id or "humanoidmaze" in env_id:
        single_observation_dim = env.unwrapped.observation_space.shape[-1]
        return locomaze_oracle_representation(goal, single_observation_dim)
    if "cube" in env_id:
        return cube_oracle_representation(goal, env.unwrapped._num_cubes)
    if "puzzle" in env_id:
        return puzzle_oracle_representation(goal, env.unwrapped._num_buttons)
    if "scene" in env_id:
        return scene_oracle_representation(
            goal,
            env.unwrapped._num_cubes,
            env.unwrapped._num_buttons,
        )
    raise ValueError(
        "Oracle goal representation currently supports AntMaze, HumanoidMaze, "
        "Cube, Puzzle, and Scene environments only."
    )


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)






def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)




def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    config = config or {}
    action_chunk_len = int(config.get('action_chunk_len', 1))
    if action_chunk_len < 1:
        raise ValueError("action_chunk_len must be positive.")
    action_dim = config.get('action_dim', None)
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        if config.get('goal_representation', 'full') == 'oracle':
            goal = _oracle_goal_representation(goal, env)
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        action_queue = []
        while not done:
            if action_chunk_len > 1:
                if len(action_queue) == 0:
                    action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
                    action = np.array(action)
                    if action_dim is None:
                        action_dim = action.shape[-1]
                    action_queue = list(action.reshape(-1, action_dim))
                    assert len(action_queue) >= action_chunk_len, (
                        f"Expected at least {action_chunk_len} chunked actions, "
                        f"got action shape {action.shape}."
                    )
                    action_queue = action_queue[:action_chunk_len]
                action = action_queue.pop(0)
            else:
                action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
                action = np.array(action)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
