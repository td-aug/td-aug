import dataclasses
from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from utils.mchc_discounts import resolve_mchc_discounts
from utils.types import (
    Array,
    DataDict,
    Statistics,
)


def _is_observation_path(path: Tuple[jax.tree_util.DictKey]) -> bool:
    return "observations" in path[0].key or "goals" in path[0].key


@partial(jax.jit, static_argnames=("method",))
def normalize(array: Array, statistics: Statistics, method="standard") -> Array:
    """Normalizes an array with given statistics."""
    if method == "standard":
        return (array - statistics.mean) / (statistics.std + 1e-10)
    elif method == "minmax":
        array = (array - statistics.min) / (
            statistics.max - statistics.min
        )  # scale to [0, 1]
        return array * 2.0 - 1.0  # scale to [-1, 1]
    elif method == "img":  # scale to [-1, 1]
        array = array / 255.0  # scale to [0, 1]
        return (array - 0.5) / 0.5  # scale to [-1, 1]
    else:
        raise ValueError(f"Unknown normalization method: {method}")


@partial(jax.jit, static_argnames=("method",))
def unnormalize(array: Array, statistics: Statistics, method="standard") -> Array:
    """Unnormalizes the array with given statistics."""
    if method == "standard":
        return array * (statistics.std + 1e-10) + statistics.mean
    elif method == "minmax":
        # clip to [-1, 1]
        array = jnp.clip(array, -1.0, 1.0)
        array = (array + 1.0) / 2.0  # scale to [0, 1]
        return array * (statistics.max - statistics.min) + statistics.min
    elif method == "img":  # scale from [-1, 1] to [0, 255]
        array = array * 0.5 + 0.5  # scale to [0, 1]
        # clip to [0, 1]
        array = jnp.clip(array, 0.0, 1.0)
        return array * 255.0  # scale to [0, 255]
    else:
        raise ValueError(f"Unknown unnormalization method: {method}")


def dict_normalize(
    data_dict: DataDict, statistics: Statistics, method="standard"
) -> DataDict:
    return jax.tree_util.tree_map_with_path(
        lambda path, x: (
            normalize(x, statistics, method) if _is_observation_path(path) else x
        ),
        data_dict,
    )


@partial(jax.jit, static_argnames=())
def dict_encode(data_dict: DataDict, encoder: Any) -> DataDict:
    def encode_leaf(path, x):
        if not _is_observation_path(path):
            return x
        key = path[0].key
        if key == "history_observations" and x.ndim >= 5:
            leading_shape = x.shape[:-3]
            flat_frames = x.reshape((-1,) + x.shape[-3:])
            encoded = jax.lax.stop_gradient(encoder(flat_frames))
            return encoded.reshape(leading_shape + encoded.shape[1:])
        return jax.lax.stop_gradient(encoder(x))

    return jax.tree_util.tree_map_with_path(
        encode_leaf,
        data_dict,
    )




def split_trajectories_by_subtasks(
    dataset: DataDict,
    metadata_path: str,
    split_every: int,
) -> Tuple[DataDict, dict]:
    """Overlay trajectory boundaries after every N completed subtasks.

    The stored transition arrays remain unchanged. This rebuilds the compact
    dataset's ``valids`` and two-marker ``terminals`` arrays from the raw
    episode terminals plus selected subtask-switch markers.
    """
    if split_every < 1:
        raise ValueError(f"split_every must be positive, got {split_every}.")

    with np.load(metadata_path) as metadata:
        required_keys = {"terminals", "subtask_terminals", "subtask_indices"}
        missing = required_keys - set(metadata.files)
        if missing:
            raise ValueError(
                f"Dataset {metadata_path} is missing subtask metadata: "
                f"{sorted(missing)}"
            )
        raw_terminals = np.asarray(metadata["terminals"], dtype=bool)
        subtask_switches = np.asarray(metadata["subtask_terminals"], dtype=bool)
        subtask_indices = np.asarray(metadata["subtask_indices"], dtype=np.int64)

    size = len(dataset["terminals"])
    if not (
        len(raw_terminals)
        == len(subtask_switches)
        == len(subtask_indices)
        == size
    ):
        raise ValueError(
            "Subtask metadata length does not match the loaded dataset: "
            f"metadata={len(raw_terminals)}, dataset={size}."
        )
    if not raw_terminals[-1]:
        raise ValueError("The final transition must be an episode terminal.")

    selected_switches = subtask_switches & (
        (subtask_indices + 1) % split_every == 0
    )
    effective_raw_terminals = raw_terminals | selected_switches

    valid_dtype = np.asarray(dataset["valids"]).dtype
    terminal_dtype = np.asarray(dataset["terminals"]).dtype
    valids = (~effective_raw_terminals).astype(valid_dtype)
    next_terminals = np.concatenate([effective_raw_terminals[1:], [True]])
    terminals = (effective_raw_terminals | next_terminals).astype(terminal_dtype)

    new_dataset = dataset.copy(
        add_or_replace={
            "valids": valids,
            "terminals": terminals,
        }
    )
    boundary_locs = np.flatnonzero(effective_raw_terminals)
    starts = np.concatenate([[0], boundary_locs[:-1] + 1])
    trajectory_lengths = boundary_locs - starts + 1
    info = {
        "split_every": split_every,
        "original_num_trajectories": int(raw_terminals.sum()),
        "available_subtask_switches": int(subtask_switches.sum()),
        "selected_subtask_splits": int(selected_switches.sum()),
        "new_num_trajectories": int(effective_raw_terminals.sum()),
        "mean_trajectory_length": float(trajectory_lengths.mean()),
        "min_trajectory_length": int(trajectory_lengths.min()),
        "max_trajectory_length": int(trajectory_lengths.max()),
    }
    return new_dataset, info


def get_size(data):
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if "valids" in self._dict:
            (self.valid_idxs,) = np.nonzero(self["valids"] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if "valids" in self._dict:
            return self.valid_idxs[
                np.random.randint(len(self.valid_idxs), size=num_idxs)
            ]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)

        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if "next_observations" not in result:
            result["next_observations"] = self._dict["observations"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        if "next_actions" not in result:
            result["next_actions"] = self._dict["actions"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(self.dataset["terminals"] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1
        assert np.isclose(
            self.config["value_p_curgoal"]
            + self.config["value_p_trajgoal"]
            + self.config["value_p_randomgoal"],
            1.0,
        )
        assert np.isclose(
            self.config["actor_p_curgoal"]
            + self.config["actor_p_trajgoal"]
            + self.config["actor_p_randomgoal"],
            1.0,
        )
        if self.config["frame_stack"] is not None:
            # Only support compact (observation-only) datasets.
            assert "next_observations" not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(
                    np.arange(self.size)
                )
                self.dataset = Dataset(
                    self.dataset.copy(dict(observations=stacked_observations))
                )
        self.norm_method = self.config.get("norm_method", "standard")
        normalize_observations = self.config.get("normalize_observations", False)
        if normalize_observations and self.norm_method == "standard":
            # compute observation statistics only when needed
            self.observation_statistics = self.compute_observation_statistics()
        elif normalize_observations and self.norm_method == "minmax":
            self.observation_statistics = self.compute_observation_minmax()
        else:
            self.observation_statistics = None

    def set_discount(self, discount):
        if isinstance(self.config, FrozenDict):
            self.config = self.config.copy({"discount": discount})
        else:
            self.config["discount"] = discount

    @property
    def chunked_valid_idxs(self):
        if self.config.get("action_chunk_len", 1) == 1:
            return None
        cached = getattr(self, "_chunked_valid_idxs_cache", None)
        if cached is not None:
            return cached

        action_chunk_len = int(self.config["action_chunk_len"])
        required_actions = action_chunk_len
        valid_idxs = []
        for initial_loc, terminal_loc in zip(
            self.initial_locs, self.terminal_locs
        ):
            # terminal_loc is the final valid action index. A start is valid
            # when all required consecutive actions remain in this trajectory.
            last_start = terminal_loc - required_actions + 1
            if initial_loc <= last_start:
                valid_idxs.append(np.arange(initial_loc, last_start + 1))

        if not valid_idxs:
            raise ValueError(
                "No trajectory is long enough for action_chunk_len="
                f"{action_chunk_len}."
            )
        self._chunked_valid_idxs_cache = np.concatenate(valid_idxs, axis=0)
        return self._chunked_valid_idxs_cache

    def compute_observation_statistics(self) -> Statistics:
        observations = self.dataset["observations"]
        mean = np.mean(observations, axis=0)
        std = np.std(observations, axis=0)
        return Statistics(mean=mean, std=std)

    def compute_observation_minmax(self) -> Statistics:
        observations = self.dataset["observations"]
        min_val = np.min(observations, axis=0)
        max_val = np.max(observations, axis=0)
        # assert that min and max are not the same
        assert not np.any(min_val == max_val), "min and max values are the same"
        return Statistics(min=min_val, max=max_val)

    def get_random_idxs(self, batch_size):
        if self.config.get("action_chunk_len", 1) == 1:
            return self.dataset.get_random_idxs(batch_size)

        # get set of valid indices.
        chunked_valid_idxs = self.chunked_valid_idxs
        return np.random.choice(chunked_valid_idxs, size=batch_size)

    def sample(self, batch_size: int, idxs=None, evaluation=False, **kwargs):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        chunk_len = self.config.get("action_chunk_len", 1)

        if idxs is None:
            idxs = self.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config["frame_stack"] is not None:
            batch["observations"] = self.get_observations(idxs)
            batch["next_observations"] = self.get_observations(idxs + chunk_len)

        # GCFBC only needs the current boundary-safe action chunk.
        if chunk_len > 1:
            idxs = np.asarray(idxs)
            action_offsets = np.arange(chunk_len)
            batch["actions"] = jax.tree_util.tree_map(
                lambda arr: arr[idxs[:, None] + action_offsets],
                self.dataset["actions"],
            )

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config["actor_p_curgoal"],
            self.config["actor_p_trajgoal"],
            self.config["actor_p_randomgoal"],
            self.config["actor_geom_sample"],
        )
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        batch["value_goals"] = self.get_goal_observations(value_goal_idxs)

        # get actions at value goal indices
        batch["value_goal_actions"] = self.dataset["actions"][value_goal_idxs]
        if self.config.get("loss_type") == "MC_HC_TD2":
            mc_offset_value_goal_idxs = np.minimum(
                value_goal_idxs + 1, final_state_idxs
            )
            batch["mc_offset_value_goals"] = self.get_observations(
                mc_offset_value_goal_idxs
            )
            batch["mc_offset_value_goal_actions"] = self.dataset["actions"][
                mc_offset_value_goal_idxs
            ]
        batch["actor_goals"] = self.get_goal_observations(actor_goal_idxs)
        # get actions at actor goal indices
        batch["actor_goal_actions"] = self.dataset["actions"][actor_goal_idxs]
        successes = (idxs == value_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if self.config["gc_negative"] else 0.0)

        # store length of the remaining trajectory from each idx
        batch["remaining_horizon"] = final_state_idxs - idxs

        if self.config["p_aug"] is not None and not evaluation:
            if np.random.rand() < self.config["p_aug"]:
                self.augment(
                    batch,
                    ["observations", "next_observations", "value_goals", "actor_goals"],
                )

        return batch

    def sample_goals(
        self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample,
    ):
        batch_size = len(idxs)

        # Handle action chunking here.
        chunk_len = self.config.get("action_chunk_len", 1)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            offsets = np.random.geometric(
                p=1 - self.config["discount"], size=batch_size
            )  # in [1, inf)

            offsets = np.maximum(offsets, chunk_len)

            middle_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform future goals, including for the maze BC recipes.
            distances = np.random.rand(batch_size)  # in [0, 1)
            middle_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)

        pick_traj = np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-6)
        goal_idxs = np.where(
            pick_traj,
            middle_goal_idxs,
            random_goal_idxs,
        )

        # Goals at the current state.
        pick_current = np.random.rand(batch_size) < p_curgoal
        goal_idxs = np.where(pick_current, idxs, goal_idxs)
        return goal_idxs

    def augment(self, batch, keys):
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int32)], axis=1
        )
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: (
                    np.array(batched_random_crop(arr, crop_froms, padding))
                    if len(arr.shape) == 4
                    else arr
                ),
                batch[key],
            )

    def get_observations(self, idxs):
        if self.config["frame_stack"] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(
                lambda arr: arr[idxs], self.dataset["observations"]
            )
        else:
            return self.get_stacked_observations(idxs)

    def get_goal_observations(self, idxs):
        if "oracle_reps" in self.dataset:
            return self.dataset["oracle_reps"][idxs]
        return self.get_observations(idxs)

    def get_stacked_observations(self, idxs):
        initial_state_idxs = self.initial_locs[
            np.searchsorted(self.initial_locs, idxs, side="right") - 1
        ]
        rets = []
        for i in reversed(range(self.config["frame_stack"])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(
                jax.tree_util.tree_map(
                    lambda arr: arr[cur_idxs], self.dataset["observations"]
                )
            )
        return jax.tree_util.tree_map(
            lambda *args: np.concatenate(args, axis=-1), *rets
        )

    def get_qpos(self, idxs):
        if self.config["frame_stack"] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset["qpos"])
        else:
            return self.get_stacked_qpos(idxs)

    def get_stacked_qpos(self, idxs):
        initial_state_idxs = self.initial_locs[
            np.searchsorted(self.initial_locs, idxs, side="right") - 1
        ]
        rets = []
        for i in reversed(range(self.config["frame_stack"])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(
                jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset["qpos"])
            )
        return jax.tree_util.tree_map(
            lambda *args: np.concatenate(args, axis=-1), *rets
        )


@dataclasses.dataclass
class OneStepHistoryGCDataset(GCDataset):
    """GCDataset that supplies boundary-safe state histories to one-step models."""

    def __post_init__(self):
        super().__post_init__()
        self.history_size = int(self.config.get("history_size", 1))
        if self.history_size < 1:
            raise ValueError("history_size must be at least 1.")
        if self.config["frame_stack"] is not None:
            raise ValueError("OneStepHistoryGCDataset does not support frame_stack.")

    def sample(self, batch_size: int, idxs=None, **kwargs):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        idxs = np.asarray(idxs)
        batch = super().sample(batch_size, idxs=idxs, **kwargs)

        initial_state_idxs = self.initial_locs[
            np.searchsorted(self.initial_locs, idxs, side="right") - 1
        ]
        offsets = np.arange(self.history_size - 1, -1, -1)
        history_idxs = np.maximum(
            idxs[:, None] - offsets[None, :], initial_state_idxs[:, None]
        )
        batch["history_observations"] = jax.tree_util.tree_map(
            lambda arr: arr[history_idxs], self.dataset["observations"]
        )
        return batch


@dataclasses.dataclass
class ChunkTDFlowGCDataset(GCDataset):
    """Single-state targets and action chunks for geometric-horizon flows."""

    def __post_init__(self):
        super().__post_init__()
        for key in ("history_size", "prediction_chunk_size"):
            if self.config.get(key, 1) != 1:
                raise ValueError("Chunk GHM supports one current state and one future state only.")
        if self.config["frame_stack"] is not None:
            raise ValueError("Chunk GHM does not support frame_stack.")
        self.action_chunk_size = int(self.config.get("action_chunk_size", 1))
        self.bootstrap_stride = int(self.config.get("bootstrap_stride", 1))
        self.bootstrap_continuation_mode = self.config.get(
            "bootstrap_continuation_mode", "recorded"
        )
        self.mchc_sampling_unit = self.config.get(
            "mchc_sampling_unit", "primitive"
        )
        if self.action_chunk_size < 1:
            raise ValueError("action_chunk_size must be at least 1.")
        if self.bootstrap_stride < 1:
            raise ValueError("bootstrap_stride must be at least 1.")
        if not self.config.get("value_geom_sample", True):
            raise ValueError("ChunkTDFlowGCDataset requires value_geom_sample=True.")
        if self.bootstrap_continuation_mode not in (
            "recorded",
            "policy_prior",
        ):
            raise ValueError(
                "bootstrap_continuation_mode must be 'recorded' "
                "or 'policy_prior'."
            )
        if self.mchc_sampling_unit == "chunk":
            resolve_mchc_discounts(self.config)
        elif self.mchc_sampling_unit != "primitive":
            raise ValueError(
                "mchc_sampling_unit must be 'primitive' or 'chunk'."
            )

        all_idxs = np.arange(self.size)
        self._final_state_idxs = self.terminal_locs[
            np.searchsorted(self.terminal_locs, all_idxs)
        ]
        full_action_chunk = (
            all_idxs + self.action_chunk_size <= self._final_state_idxs
        )
        has_bootstrap_state = (
            all_idxs + self.bootstrap_stride <= self._final_state_idxs
        )
        source_valid = full_action_chunk & has_bootstrap_state
        if "valids" in self.dataset:
            source_valid &= np.asarray(self.dataset["valids"]) > 0
        self._chunk_td_valid_idxs = all_idxs[source_valid]
        if len(self._chunk_td_valid_idxs) == 0:
            raise ValueError(
                "No source state has a complete action chunk and bootstrap state."
            )


    def get_random_idxs(self, batch_size):
        if (
            self.action_chunk_size == 1
            and self.bootstrap_stride == 1
            and self.bootstrap_continuation_mode == "recorded"
        ):
            return super().get_random_idxs(batch_size)
        return np.random.choice(self._chunk_td_valid_idxs, size=batch_size)

    def _action_chunks_at(self, idxs):
        idxs = np.asarray(idxs)
        final_state_idxs = self._final_state_idxs[idxs]
        raw_action_idxs = idxs[:, None] + np.arange(self.action_chunk_size)[None]
        action_valids = raw_action_idxs < final_state_idxs[:, None]
        action_idxs = np.minimum(raw_action_idxs, final_state_idxs[:, None])
        actions = np.array(self.dataset["actions"][action_idxs], copy=True)
        actions[~action_valids] = 0.0
        return actions, action_valids.astype(np.float32)

    def _bootstrap_inputs(self, idxs):
        idxs = np.asarray(idxs)
        observations = self.get_observations(idxs)
        actions, action_valids = self._action_chunks_at(idxs)
        # The training loop replaces marked chunks with fresh prior samples.
        use_prior = np.full(idxs.shape, self.bootstrap_continuation_mode == "policy_prior")
        return observations, actions, action_valids, use_prior.astype(np.float32), idxs

    def sample(self, batch_size: int, idxs=None, **kwargs):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        idxs = np.asarray(idxs)
        batch = super().sample(batch_size, idxs=idxs, **kwargs)
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        bootstrap_idxs = np.minimum(idxs + self.bootstrap_stride, final_state_idxs)
        (
            batch["bootstrap_observations"],
            batch["bootstrap_actions"],
            batch["bootstrap_action_valids"],
            batch["bootstrap_uses_action_prior"],
            batch["bootstrap_state_idxs"],
        ) = self._bootstrap_inputs(bootstrap_idxs)

        # Clamp geometric futures at the terminal state.
        remaining = final_state_idxs - idxs
        max_start_offsets = np.maximum(remaining, 1)
        local_sampling_discount = float(self.config["discount"])
        if self.mchc_sampling_unit == "chunk":
            local_sampling_discount, _ = resolve_mchc_discounts(self.config)
        geometric_offsets = np.random.geometric(
            p=1.0 - local_sampling_discount, size=batch_size
        )
        mc_offset = int(self.config.get("loss_type") == "MC_HC_TD2")
        if self.mchc_sampling_unit == "chunk":
            sampled_start_offsets = self.bootstrap_stride * (
                geometric_offsets + mc_offset
            )
        else:
            sampled_start_offsets = (
                geometric_offsets + mc_offset + self.bootstrap_stride - 1
            )
        start_offsets = np.minimum(sampled_start_offsets, max_start_offsets)
        future_bootstrap_idxs = np.minimum(idxs + start_offsets, final_state_idxs)
        batch["future_chunk_start_offsets"] = start_offsets.astype(np.int32)
        (
            batch["future_observations"],
            batch["future_chunk_actions"],
            batch["future_chunk_action_valids"],
            batch["future_bootstrap_uses_action_prior"],
            batch["future_bootstrap_state_idxs"],
        ) = self._bootstrap_inputs(future_bootstrap_idxs)

        batch["condition_actions"], batch["condition_action_valids"] = self._action_chunks_at(idxs)
        return batch


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config["frame_stack"] is not None:
            batch["observations"] = self.get_observations(idxs)
            batch["next_observations"] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        batch["value_goals"] = self.get_goal_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if self.config["gc_negative"] else 0.0)

        # Set low-level actor goals.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(
            idxs + self.config["subgoal_steps"], final_state_idxs
        )
        batch["low_actor_goals"] = self.get_goal_observations(low_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        if self.config["actor_geom_sample"]:
            offsets = np.random.geometric(
                p=1 - self.config["discount"], size=batch_size
            )  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        high_traj_target_idxs = np.minimum(
            idxs + self.config["subgoal_steps"], high_traj_goal_idxs
        )

        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(
            idxs + self.config["subgoal_steps"], final_state_idxs
        )

        pick_random = np.random.rand(batch_size) < self.config["actor_p_randomgoal"]
        high_goal_idxs = np.where(
            pick_random, high_random_goal_idxs, high_traj_goal_idxs
        )
        high_target_idxs = np.where(
            pick_random, high_random_target_idxs, high_traj_target_idxs
        )

        batch["high_actor_goals"] = self.get_goal_observations(high_goal_idxs)
        batch["high_actor_targets"] = self.get_goal_observations(high_target_idxs)
        # Value evaluation needs the full state, not its projected subgoal.
        batch["high_actor_next_observations"] = self.get_observations(high_target_idxs)

        if self.config["p_aug"] is not None and not evaluation:
            if np.random.rand() < self.config["p_aug"]:
                self.augment(
                    batch,
                    [
                        "observations",
                        "next_observations",
                        "value_goals",
                        "low_actor_goals",
                        "high_actor_goals",
                        "high_actor_targets",
                    ],
                )

        if "oracle_reps" not in self.dataset:
            batch["high_actor_next_observations"] = batch["high_actor_targets"]

        return batch


@dataclasses.dataclass
class AugGCDataset:
    """Dataset class for goal-conditioned RL. Basically the same as `GCDataset`, but introduces utilities for generating augmented goals.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_p_auggoal: Probability of using an augmented goal (given by `goal_sample_fn`) as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_p_auggoal: Probability of using an augmented goal (given by `goal_sample_fn`) as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        goal_sample_fn: Goal sampling function, taking in observations and actions and returning goals.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    gen_config: Any
    gen_model: Any
    preprocess_frame_stack: bool = True
    gen_ae: Any = None

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(self.dataset["terminals"] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1
        assert np.isclose(
            self.config["value_p_curgoal"]
            + self.config["value_p_trajgoal"]
            + self.config["value_p_randomgoal"]
            + self.config["value_p_auggoal"],
            1.0,
        )
        assert np.isclose(
            self.config["actor_p_curgoal"]
            + self.config["actor_p_trajgoal"]
            + self.config["actor_p_randomgoal"]
            + self.config["actor_p_auggoal"],
            1.0,
        )
        if self.config["frame_stack"] is not None:
            # Only support compact (observation-only) datasets.
            assert "next_observations" not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(
                    np.arange(self.size)
                )
                self.dataset = Dataset(
                    self.dataset.copy(dict(observations=stacked_observations))
                )

        self.norm_method = self.gen_config.get(
            "norm_method", self.config.get("norm_method", "standard")
        )
        normalize_observations = self.gen_config.get("normalize_observations", False)
        if normalize_observations and self.norm_method == "standard":
            self.observation_statistics = self.compute_observation_statistics()
        elif normalize_observations and self.norm_method == "minmax":
            self.observation_statistics = self.compute_observation_minmax()
        else:
            self.observation_statistics = None

    def set_discount(self, discount):
        if isinstance(self.config, FrozenDict):
            self.config = self.config.copy({"discount": discount})
        else:
            self.config["discount"] = discount

    def compute_observation_statistics(self) -> Statistics:
        observations = self.dataset["observations"]
        mean = np.mean(observations, axis=0)
        std = np.std(observations, axis=0)
        return Statistics(mean=mean, std=std)

    def compute_observation_minmax(self) -> Statistics:
        observations = self.dataset["observations"]
        return Statistics(
            min=np.min(observations, axis=0), max=np.max(observations, axis=0)
        )

    def generate_goals(self, obs, actions):
        if self.gen_ae is not None:
            if self.gen_config["normalize_observations"]:
                raise ValueError(
                    "AE-latent augmented goal generation does not support "
                    "gen_config['normalize_observations']; the raw-image policy "
                    "dataset statistics do not apply to AE latents."
                )
            obs = np.asarray(jax.device_get(self.gen_ae.encode(obs)))
        elif self.gen_config["normalize_observations"]:
            obs = normalize(obs, self.observation_statistics, self.norm_method)

        self.gen_model, model_samples = self.gen_model.sample(obs, actions)
        if self.gen_ae is not None:
            model_samples = np.asarray(
                jax.device_get(self.gen_ae.decode(model_samples))
            )
            if not self.gen_ae.config.get("scale_out", True):
                model_samples = (model_samples * 255.0).clip(0, 255)
        elif self.gen_config["normalize_observations"]:
            model_samples = unnormalize(
                model_samples, self.observation_statistics, self.norm_method
            )

        return model_samples

    def sample(self, batch_size, idxs=None, evaluation=False, aug_goals=True):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config["frame_stack"] is not None:
            batch["observations"] = self.get_observations(idxs)
            batch["next_observations"] = self.get_observations(idxs + 1)

        value_goal_idxs, value_aug_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_auggoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        actor_goal_idxs, actor_aug_goal_idxs = self.sample_goals(
            idxs,
            self.config["actor_p_curgoal"],
            self.config["actor_p_trajgoal"],
            self.config["actor_p_auggoal"],
            self.config["actor_p_randomgoal"],
            self.config["actor_geom_sample"],
        )

        batch["value_goals"] = self.get_observations(value_goal_idxs)
        batch["actor_goals"] = self.get_observations(actor_goal_idxs)

        # do the goal generation now.
        if aug_goals:
            if len(value_aug_goal_idxs) > 0:
                value_aug_observations = batch["observations"][value_aug_goal_idxs]
                value_aug_actions = batch["actions"][value_aug_goal_idxs]
                value_aug_goals = self.generate_goals(
                    value_aug_observations, value_aug_actions
                )
                batch["value_goals"][value_aug_goal_idxs] = value_aug_goals

            if len(actor_aug_goal_idxs) > 0:
                actor_aug_observations = batch["observations"][actor_aug_goal_idxs]
                actor_aug_actions = batch["actions"][actor_aug_goal_idxs]
                actor_aug_goals = self.generate_goals(
                    actor_aug_observations, actor_aug_actions
                )
                batch["actor_goals"][actor_aug_goal_idxs] = actor_aug_goals

        # we only succeed when the TRUE indices are equal to the value goal indices.
        successes = (idxs == value_goal_idxs).astype(float)
        successes[value_aug_goal_idxs] = 0.0

        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if self.config["gc_negative"] else 0.0)

        if self.config["p_aug"] is not None and not evaluation:
            if np.random.rand() < self.config["p_aug"]:
                self.augment(
                    batch,
                    ["observations", "next_observations", "value_goals", "actor_goals"],
                )

        return batch

    def sample_goals(
        self, idxs, p_curgoal, p_trajgoal, p_auggoal, p_randomgoal, geom_sample
    ):
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            offsets = np.random.geometric(
                p=1 - self.config["discount"], size=batch_size
            )  # in [1, inf)
            middle_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)  # in [0, 1)
            middle_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        goal_idxs = np.where(
            np.random.rand(batch_size)
            < p_trajgoal / (1.0 - p_curgoal - p_auggoal + 1e-6),
            middle_goal_idxs,
            random_goal_idxs,
        )

        # Goals at the current state. Choose a set of them to be augmented later.
        cur_or_aug = np.random.rand(batch_size) < p_curgoal + p_auggoal
        num_cur_or_aug = np.sum(cur_or_aug)
        if p_curgoal + p_auggoal == 0:
            num_aug = 0
        else:
            num_aug = int(num_cur_or_aug * (p_auggoal / (p_curgoal + p_auggoal)))

        cur_or_aug_idxs = np.nonzero(cur_or_aug > 0)[0]
        aug_idxs = np.random.choice(cur_or_aug_idxs, size=num_aug, replace=False)

        goal_idxs = np.where(cur_or_aug, idxs, goal_idxs)
        return goal_idxs, aug_idxs

    def augment(self, batch, keys):
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int32)], axis=1
        )
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: (
                    np.array(batched_random_crop(arr, crop_froms, padding))
                    if len(arr.shape) == 4
                    else arr
                ),
                batch[key],
            )

    def get_observations(self, idxs):
        if self.config["frame_stack"] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(
                lambda arr: arr[idxs], self.dataset["observations"]
            )
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        initial_state_idxs = self.initial_locs[
            np.searchsorted(self.initial_locs, idxs, side="right") - 1
        ]
        rets = []
        for i in reversed(range(self.config["frame_stack"])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(
                jax.tree_util.tree_map(
                    lambda arr: arr[cur_idxs], self.dataset["observations"]
                )
            )
        return jax.tree_util.tree_map(
            lambda *args: np.concatenate(args, axis=-1), *rets
        )
