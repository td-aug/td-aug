import collections
import os
import platform
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if terminated or truncated:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self.unwrapped, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.unwrapped.get_normalized_score(info["episode"]["return"])
                    * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if "goal" in info:
            info["goal"] = np.concatenate([info["goal"]] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.get_observation(), reward, terminated, truncated, info


def setup_egl():
    """Set up EGL for rendering."""
    if "mac" in platform.platform():
        # macOS doesn't support EGL.
        pass
    else:
        os.environ["MUJOCO_GL"] = "egl"
        if "SLURM_STEP_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]


def get_dataset(
    dataset_path,
    ob_dtype=np.float32,
    action_dtype=np.float32,
    get_qpos=False,
    flatten_obs=False,
    get_val_dataset=True,
    same_eval=False,
):
    """Load OGBench dataset.

    Args:
        dataset_path: Path to the dataset file.
        ob_dtype: dtype for observations.
        action_dtype: dtype for actions.

    Returns:
        Tuple of training and validation datasets.
    """
    train_path = dataset_path
    train_dataset = dict()
    val_path = None
    val_dataset = None
    if get_val_dataset:
        # use same path for val if train already has -val
        if same_eval:
            val_path = train_path
        elif "-val" in dataset_path:
            val_path = dataset_path
        else:
            val_path = dataset_path.replace(".npz", "-val.npz")
        val_dataset = dict()

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        if path is None:
            continue
        file = np.load(path)

        if get_qpos:
            classes = ["observations", "actions", "qpos", "terminals"]
        else:
            classes = ["observations", "actions", "terminals"]
        # Preserve optional transition metadata used by specialized offline
        # datasets. Core OGBench datasets do not need these fields.
        for optional_key in ("task_ids",):
            if optional_key in file:
                classes.append(optional_key)

        for k in classes:
            if k == "observations":
                dtype = ob_dtype
                dataset[k] = file[k][...].astype(dtype)
                if flatten_obs:
                    dataset[k] = dataset[k].reshape(dataset[k].shape[0], -1)
            elif k == "actions" or k == "qpos":
                dtype = action_dtype
                dataset[k] = file[k][...].astype(dtype)
            elif k == "task_ids":
                dataset[k] = file[k][...].astype(np.int32)
            else:
                dtype = np.float32
                dataset[k] = file[k][...].astype(dtype)

        # At this point, we have:
        # terminals: [0, 0, ..., 0, 0, 1, 0, 0, ..., 0, 0, 1, 0, 0, ..., 0, 0, 1]
        #             |<--trajectory-->|  |<--trajectory-->|  |<- trajectory-->|

        # Since the dataset doesn't contain next_observations, we need to invalidate the last state of each trajectory
        # so that we can safely get next_observations[t] by using observations[t + 1].
        dataset["valids"] = 1.0 - dataset["terminals"]
        new_terminals = np.concatenate([dataset["terminals"][1:], [1.0]])
        dataset["terminals"] = np.minimum(dataset["terminals"] + new_terminals, 1.0)

        # Now, we have the following:
        # terminals: [0, 0, ..., 0, 1, 1, 0, 0, ..., 0, 1, 1, 0, 0, ..., 0, 1, 1]
        # valids   : [1, 1, ..., 1, 1, 0, 1, 1, ..., 1, 0, 1, 1, 1, ..., 1, 1, 0]
        #             |<--trajectory-->|  |<--trajectory-->|  |<--trajectory-->|
    if get_val_dataset:
        return Dataset.create(**train_dataset), Dataset.create(**val_dataset)
    else:
        return Dataset.create(**train_dataset)


def make_env_and_dataset(
    env_name,
    dataset_path=None,
    frame_stack=None,
    precomputed_features=False,
    implemtented=False,
    flatten_obs=False,
    same_eval=False,
):
    """Make environment and dataset.

    Args:
        env_name: Name of the environment.
        dataset_path: Path to the dataset file.
        frame_stack: Number of frames to stack.
    """
    setup_egl()

    if "visual" in env_name:
        assert implemtented, "Visual environment need precomputed recheck!"

    if (
        "pointmaze" in env_name
        or "antmaze" in env_name
        or "humanoidmaze" in env_name
    ):
        # OGBench Locomotion environments.
        import ogbench.locomaze  # noqa

        env = gymnasium.make(env_name)
        ob_dtype = (
            np.uint8
            if ("visual" in env_name and not precomputed_features)
            else np.float32
        )
        train_dataset, val_dataset = get_dataset(
            dataset_path,
            ob_dtype=ob_dtype,
            get_qpos=True if "visual" in env_name else False,
            flatten_obs=flatten_obs,
            same_eval=same_eval,
        )
    elif "cube" in env_name or "scene" in env_name or "puzzle" in env_name:
        # OGBench Manipulation environments.
        import ogbench.manipspace  # noqa

        env = gymnasium.make(env_name)
        ob_dtype = (
            np.uint8
            if ("visual" in env_name and not precomputed_features)
            else np.float32
        )
        train_dataset, val_dataset = get_dataset(
            dataset_path, ob_dtype=ob_dtype, flatten_obs=flatten_obs, same_eval=same_eval
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    if val_dataset is not None and val_dataset.size == 0:
        val_dataset = None

    env.reset()

    return env, train_dataset, val_dataset

def make_dataset(
    env_name,
    dataset_path=None,
    precomputed_features=False,
    flatten_obs=False,
):
    """Make environment and dataset.

    Args:
        env_name: Name of the environment.
        dataset_path: Path to the dataset file.
    """

    if (
        "pointmaze" in env_name
        or "antmaze" in env_name
        or "humanoidmaze" in env_name
    ):
        # OGBench Locomotion environments.
        ob_dtype = (
            np.uint8
            if ("visual" in env_name and not precomputed_features)
            else np.float32
        )
        dataset = get_dataset(
            dataset_path,
            ob_dtype=ob_dtype,
            get_qpos=True if "visual" in env_name else False,
            flatten_obs=flatten_obs,
            get_val_dataset=False,
        )
    elif "cube" in env_name or "scene" in env_name or "puzzle" in env_name:
        # OGBench Manipulation environments.
        ob_dtype = (
            np.uint8
            if ("visual" in env_name and not precomputed_features)
            else np.float32
        )
        dataset = get_dataset(
            dataset_path, ob_dtype=ob_dtype, flatten_obs=flatten_obs, get_val_dataset=False
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return dataset



def make_online_env(env_name):
    """Make online environment.

    Args:
        env_name: Name of the environment.
    """
    setup_egl()

    if "online-ant" in env_name or "online-humanoid" in env_name:
        import ogbench.online_locomotion  # noqa

        # Manually recognize the '-xy' suffix, which indicates that the environment should be wrapped with a
        # directional locomotion wrapper.
        if "-xy" in env_name:
            env_name = env_name.replace("-xy", "")
            apply_xy_wrapper = True
        else:
            apply_xy_wrapper = False

        if "humanoid" in env_name:
            env = gymnasium.make(
                env_name, render_mode="rgb_array", height=200, width=200, camera_id=0
            )
        else:
            env = gymnasium.make(
                env_name, render_mode="rgb_array", height=200, width=200
            )

        if apply_xy_wrapper:
            # Apply the directional locomotion wrapper.
            from ogbench.online_locomotion.wrappers import (
                DMCHumanoidXYWrapper,
                GymXYWrapper,
            )

            if "humanoid" in env_name:
                env = DMCHumanoidXYWrapper(env, resample_interval=200)
            else:
                env = GymXYWrapper(env, resample_interval=100)

        env = EpisodeMonitor(env)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return env


def make_vec_env(env_name, num_envs, **kwargs):
    from gymnasium.vector import SyncVectorEnv

    envs = [lambda: make_online_env(env_name, **kwargs) for _ in range(num_envs)]
    env = SyncVectorEnv(envs)
    return env
