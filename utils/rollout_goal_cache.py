"""CPU-RAM-only future trajectories for frozen one-step policy augmentation."""
import time

import numpy as np

from utils.datasets import normalize, unnormalize


class RolloutGoalCache:
    def __init__(self, goals, discount, seed):
        if goals.ndim != 3 or goals.shape[1] < 1:
            raise ValueError("Expected cache [N, T, D] with T >= 1")
        if not 0 <= discount < 1:
            raise ValueError("Cache discount must be in [0, 1)")
        self.goals = goals
        self.discount = float(discount)
        self.rng = np.random.default_rng(seed)

    def augment(self, batch, indices, n_augment):
        indices = np.asarray(indices)
        size = len(batch["observations"])
        if indices.shape != (size,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("One integer source index is required per batch example")
        if np.any(indices < 0) or np.any(indices >= len(self.goals)):
            raise ValueError("Source index outside rollout cache")
        if not 0 <= n_augment <= size:
            raise ValueError("n_augment outside batch size")
        result = dict(batch)
        goals = np.array(batch["actor_goals"], copy=True)
        if goals.shape != (size, self.goals.shape[-1]):
            raise ValueError("Policy goal shape does not match cached raw observations")
        selected = self.rng.choice(size, size=n_augment, replace=False)
        offsets = self.rng.geometric(1 - self.discount, size=n_augment)
        offsets = np.minimum(offsets, self.goals.shape[1])
        goals[selected] = self.goals[indices[selected], offsets - 1]
        mask = np.zeros(size, dtype=bool)
        mask[selected] = True
        result.update(actor_goals=goals, aug_mask=mask)
        return result


def build_rollout_goal_cache(model, source_dataset, config, stat_dataset,
                            batch_size=2048, seed=0, progress=None):
    """Generate one trajectory per raw dataset row; stream GPU batches to CPU."""
    if config.get("agent_name") != "onestep":
        raise ValueError("RAM rollout caching currently supports agents/onestep.py only")
    steps = int(config["sample_max_steps"])
    discount = float(config["discount"])
    if batch_size < 1 or steps < 1 or not 0 <= discount < 1:
        raise ValueError("Require positive batch/cap and discount in [0, 1)")
    observations = source_dataset["observations"]
    actions = source_dataset["actions"]
    if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions):
        raise ValueError("Cache sources must be aligned vector states and single actions")
    size, dim = observations.shape
    if size == 0:
        raise ValueError("Cannot cache an empty dataset")
    goals = np.empty((size, steps, dim), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, size, batch_size):
        end = min(start + batch_size, size)
        # Pad only the last batch to avoid recompiling the full rollout.
        idx = np.minimum(np.arange(start, start + batch_size), size - 1)
        obs, act = observations[idx], actions[idx]
        if config["normalize_observations"]:
            obs = normalize(obs, stat_dataset.observation_statistics, stat_dataset.norm_method)
        model, predicted = model.sample_rollout_jnp(obs, act, max_steps=steps)
        if config["normalize_observations"]:
            predicted = unnormalize(predicted, stat_dataset.observation_statistics,
                                    stat_dataset.norm_method)
        host = np.asarray(predicted, dtype=np.float32)[:end-start]
        if host.shape != (end-start, steps, dim) or not np.isfinite(host).all():
            raise ValueError(f"Invalid generated rollout in rows {start}:{end}")
        goals[start:end] = host
        if progress is not None:
            progress(end, size, time.perf_counter() - started)
    return RolloutGoalCache(goals, discount, seed), model
