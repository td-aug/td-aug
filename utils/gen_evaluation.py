from typing import Any, Optional, Tuple, Union

import jax
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import jax.numpy as jnp

from utils.datasets import normalize, unnormalize
from utils.flow_utils import sample_model_states


def _set_array_indices(array, indices, values):
    if hasattr(array, "at"):
        return array.at[indices].set(jnp.asarray(values))
    array[indices] = np.asarray(values)
    return array


def _sample_geometric_offsets(discount, size, max_steps=None):
    offsets = np.random.geometric(p=1.0 - discount, size=size)
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")
        offsets = np.minimum(offsets, max_steps)
    return offsets


def _source_actions_for_model(dataset, source_indices, config):
    """Return current actions or aligned fixed chunks for a chunk model."""
    source_indices = np.asarray(source_indices)
    actions = np.asarray(dataset.dataset["actions"])
    if config.get("agent_name") != "chunk_tdflow":
        return actions[source_indices]

    action_chunk_size = int(config["action_chunk_size"])
    if action_chunk_size == 1:
        return actions[source_indices]
    terminal_indices = dataset.terminal_locs[
        np.searchsorted(dataset.terminal_locs, source_indices)
    ]
    raw_indices = source_indices[:, None] + np.arange(action_chunk_size)[None, :]
    valid = raw_indices < terminal_indices[:, None]
    clipped_indices = np.minimum(raw_indices, terminal_indices[:, None])
    chunks = np.array(actions[clipped_indices], copy=True)
    chunks[~valid] = 0.0
    return chunks


def supply_sample_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def _get_model_samples_vae(
    model,
    source_observations: np.ndarray,
    num_samples_per_state: int,
):
    """Gets model samples from a VAE."""

    # NOTE: this method can be `jax.jit` if we return jax arrays instead of numpy arrays in sampling.
    # source observations shape: [B, observation_dim]
    # source actions shape: [B, action_dim]
    B = source_observations.shape[0]
    obs = np.expand_dims(source_observations, 1)  # (B,1,...)
    obs = np.tile(
        obs,
        (1, num_samples_per_state) + (1,) * (obs.ndim - 2),
    )  # (B,N,...)
    obs = obs.reshape((B * num_samples_per_state,) + obs.shape[2:])  # (B*N,...)

    # sample from model -> shape: [B * N, observation_dim]
    sample_fn = supply_sample_rng(
        model.sample, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))
    )
    model_samples = sample_fn(obs)
    return model_samples


def _dataset_xy_targets(dataset):
    if "qpos" in dataset.dataset:
        return dataset.dataset["qpos"][:, :2]
    return dataset.dataset["observations"][:, :2]


def _extract_xy_from_samples(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim < 2 or samples.shape[-1] < 2:
        raise ValueError(
            f"Expected samples with trailing dimension >= 2 for XY extraction, got shape {samples.shape}"
        )
    return samples[..., :2]


def _ott_emd_cost_batch(
    true_samples: jnp.ndarray,
    model_samples: jnp.ndarray,
    epsilon: float = 1e-2,
) -> jnp.ndarray:
    # Training-time sample plots do not require the optional EMD dependency.
    from ott.geometry import costs as ott_costs
    from ott.geometry import pointcloud as ott_pointcloud
    from ott.problems.linear import linear_problem as ott_linear_problem
    from ott.solvers.linear import sinkhorn as ott_sinkhorn

    def one_pair(x, y):
        geom = ott_pointcloud.PointCloud(
            x, y, epsilon=epsilon, cost_fn=ott_costs.Euclidean()
        )
        prob = ott_linear_problem.LinearProblem(geom)
        solver = ott_sinkhorn.Sinkhorn()
        with jax.numpy_rank_promotion("allow"):
            sol = solver(prob)
        return sol.reg_ot_cost

    return jax.vmap(one_pair)(true_samples, model_samples)


def compute_emd_xy(
    model,
    dataset,
    config,
    num_source_states: int = 128,
    num_samples_per_source_state: int = 128,
    discount: Optional[float] = None,
    sample_max_steps: Optional[int] = None,
    epsilon: float = 1e-2,
    num_sinkhorn_iters: int = 100,
    source_encoder: Any = None,
    sample_decoder: Any = None,
    stat_dataset=None,
):
    """Computes an empirical EMD estimate in XY space.

    The estimate is averaged over source states. For each source trajectory start state:
    - sample geometric-horizon future XY states from the dataset
    - sample the same number of futures from the model
    - compute a regularized OT cost between the two XY point clouds
    """

    stat_dataset = dataset if stat_dataset is None else stat_dataset
    observations = dataset.dataset["observations"]
    xy_targets = _dataset_xy_targets(dataset)
    episode_starts = dataset.initial_locs
    episode_ends = dataset.terminal_locs
    lengths = episode_ends - episode_starts + 1
    valid_indices = np.where(lengths > 1)[0]
    episode_starts = episode_starts[valid_indices]
    episode_ends = episode_ends[valid_indices]
    lengths = lengths[valid_indices]
    if len(episode_starts) == 0:
        raise ValueError("No valid trajectories with length > 1 for EMD_XY evaluation.")

    num_source_states = min(num_source_states, len(episode_starts))
    plot_indices = np.random.choice(len(episode_starts), size=num_source_states, replace=False)
    starts = episode_starts[plot_indices]
    ends = episode_ends[plot_indices]
    traj_lengths = lengths[plot_indices]

    if discount is None:
        discount = float(config.get("discount", 0.99))

    true_samples = []
    for start, traj_length in zip(starts, traj_lengths):
        offsets = _sample_geometric_offsets(
            discount, size=num_samples_per_source_state, max_steps=sample_max_steps
        )
        offsets = np.clip(offsets, 1, traj_length - 1)
        true_samples.append(xy_targets[start + offsets])
    true_samples = np.asarray(true_samples)

    source_obs = np.stack([observations[idx] for idx in starts])
    source_actions = _source_actions_for_model(dataset, starts, config)
    model_source = source_obs
    if config["normalize_observations"]:
        model_source = normalize(
            model_source,
            stat_dataset.observation_statistics,
            stat_dataset.norm_method,
        )
    if source_encoder is not None:
        model_source = np.asarray(jax.device_get(source_encoder(model_source)))

    model, model_samples = sample_model_states(
        model,
        model_source,
        source_actions,
        num_samples_per_state=num_samples_per_source_state,
        sample_discount=discount,
        sample_max_steps=sample_max_steps,
    )
    model_samples = np.asarray(model_samples).reshape(
        num_source_states, num_samples_per_source_state, *np.asarray(model_samples).shape[1:]
    )

    if sample_decoder is not None:
        flat_samples = model_samples.reshape(
            num_source_states * num_samples_per_source_state, *model_samples.shape[2:]
        )
        model_xy = np.asarray(jax.device_get(sample_decoder(flat_samples))).reshape(
            num_source_states, num_samples_per_source_state, -1
        )
    else:
        if config["normalize_observations"]:
            flat_samples = model_samples.reshape(
                num_source_states * num_samples_per_source_state, *model_samples.shape[2:]
            )
            flat_samples = unnormalize(
                flat_samples,
                stat_dataset.observation_statistics,
                stat_dataset.norm_method,
            )
            model_samples = np.asarray(flat_samples).reshape(
                num_source_states, num_samples_per_source_state, -1
            )
        model_xy = _extract_xy_from_samples(model_samples)

    emd_per_state = np.asarray(
        jax.device_get(
            _ott_emd_cost_batch(
                jnp.asarray(true_samples),
                jnp.asarray(model_xy),
                epsilon=epsilon,
            )
        )
    )
    return model, {
        "emd_xy": float(np.mean(emd_per_state)),
        "emd_xy_std": float(np.std(emd_per_state)),
        "emd_xy_per_state": emd_per_state,
        "emd_xy_backend": "ott",
    }


def _cube_position_coordinates(samples):
    """Extract the scaled XYZ coordinates for every cube."""
    samples = np.asarray(samples)
    observation_dim = samples.shape[-1]
    if observation_dim <= 19 or (observation_dim - 19) % 9 != 0:
        raise ValueError(
            "Cube observations must contain 19 proprioceptive dimensions "
            f"followed by 9 dimensions per cube, got {observation_dim}."
        )
    num_cubes = (observation_dim - 19) // 9
    position_indices = np.concatenate(
        [np.arange(19 + 9 * cube_idx, 22 + 9 * cube_idx) for cube_idx in range(num_cubes)]
    )
    return samples[..., position_indices]


def _cube_source_candidate_idxs(dataset, config, source_mode="random"):
    """Return boundary-safe Cube source indices for the requested source mode."""
    if source_mode not in ("random", "initial"):
        raise ValueError(
            "Cube source_mode must be 'random' or 'initial', got "
            f"{source_mode!r}."
        )

    action_chunk_size = int(config["action_chunk_size"])
    valid_idxs = np.asarray(
        getattr(dataset.dataset, "valid_idxs", np.arange(dataset.size))
    )
    terminal_idxs = dataset.terminal_locs[
        np.searchsorted(dataset.terminal_locs, valid_idxs)
    ]
    has_full_action_chunk = terminal_idxs - valid_idxs >= action_chunk_size
    candidate_idxs = valid_idxs[has_full_action_chunk]
    if source_mode == "initial":
        candidate_idxs = candidate_idxs[
            np.isin(candidate_idxs, dataset.initial_locs)
        ]
    if len(candidate_idxs) == 0:
        raise ValueError(
            "No cube source transitions satisfy the action boundaries "
            f"for source_mode={source_mode!r}."
        )
    return candidate_idxs


def compute_cube_chunk_emd(
    model,
    dataset,
    config,
    num_source_states: int = 50,
    num_samples_per_source_state: int = 64,
    epsilon: float = 1e-2,
    stat_dataset=None,
    source_mode: str = "random",
    cube_position_tolerance: float = 0.4,
):
    """Compute per-source EMD for empirical and generated cube future states.

    Sources have a complete action chunk. EMD compares future states per source
    and is then summarized across source states.
    """
    if not hasattr(model, "sample"):
        raise TypeError(
            "Cube EMD requires sample()."
        )
    if num_source_states < 1 or num_samples_per_source_state < 1:
        raise ValueError("Cube EMD source and sample counts must be positive.")

    stat_dataset = dataset if stat_dataset is None else stat_dataset
    chunk_size = 1
    candidate_idxs = _cube_source_candidate_idxs(dataset, config, source_mode)

    num_source_states = min(num_source_states, len(candidate_idxs))
    source_idxs = np.random.choice(
        candidate_idxs, size=num_source_states, replace=False
    )
    source_batch = dataset.sample(
        num_source_states, idxs=source_idxs, evaluation=True
    )
    repeated_source_idxs = np.repeat(
        source_idxs, num_samples_per_source_state
    )
    empirical_batch = dataset.sample(
        len(repeated_source_idxs),
        idxs=repeated_source_idxs,
        evaluation=True,
    )
    empirical = np.asarray(
        empirical_batch["future_observations"]
    ).reshape(
        num_source_states,
        num_samples_per_source_state,
        chunk_size,
        -1,
    )

    observations = np.asarray(source_batch["observations"])
    actions = np.asarray(
        source_batch.get("condition_actions", source_batch["actions"])
    )
    if config["normalize_observations"]:
        observations = normalize(
            observations,
            stat_dataset.observation_statistics,
            stat_dataset.norm_method,
        )
    model, generated = model.sample(
        np.repeat(observations, num_samples_per_source_state, axis=0),
        np.repeat(actions, num_samples_per_source_state, axis=0),
    )
    generated = np.asarray(generated)
    if config["normalize_observations"]:
        generated = unnormalize(
            generated,
            stat_dataset.observation_statistics,
            stat_dataset.norm_method,
        )
    generated = generated.reshape(
        num_source_states,
        num_samples_per_source_state,
        chunk_size,
        -1,
    )

    source_terminal_idxs = dataset.terminal_locs[
        np.searchsorted(dataset.terminal_locs, source_idxs)
    ]
    terminal_observations = np.asarray(dataset.dataset["observations"])[
        source_terminal_idxs
    ]

    def completion_mass(samples):
        sample_positions = _cube_position_coordinates(samples)
        target_positions = _cube_position_coordinates(
            terminal_observations
        )
        num_cubes = target_positions.shape[-1] // 3
        sample_positions = sample_positions.reshape(
            *sample_positions.shape[:-1], num_cubes, 3
        )
        target_positions = target_positions.reshape(
            num_source_states, num_cubes, 3
        )[:, None, None]
        cube_distances = np.linalg.norm(
            sample_positions - target_positions, axis=-1
        )
        return np.all(cube_distances <= cube_position_tolerance, axis=-1)

    empirical_completion = completion_mass(empirical)
    generated_completion = completion_mass(generated)
    empirical_completion_mass = float(empirical_completion.mean())
    generated_completion_mass = float(generated_completion.mean())

    def summarize(true_samples, model_samples):
        true_by_position = np.transpose(true_samples, (0, 2, 1, 3))
        model_by_position = np.transpose(model_samples, (0, 2, 1, 3))
        costs = np.asarray(
            jax.device_get(
                _ott_emd_cost_batch(
                    jnp.asarray(
                        true_by_position.reshape(
                            num_source_states * chunk_size,
                            num_samples_per_source_state,
                            -1,
                        )
                    ),
                    jnp.asarray(
                        model_by_position.reshape(
                            num_source_states * chunk_size,
                            num_samples_per_source_state,
                            -1,
                        )
                    ),
                    epsilon=epsilon,
                )
            )
        ).reshape(num_source_states, chunk_size)
        per_source = costs.mean(axis=1)
        return {
            "mean": float(per_source.mean()),
            "std": float(per_source.std()),
            "per_position": costs.mean(axis=0),
            "per_source": per_source,
        }

    raw = summarize(empirical, generated)
    cube_position = summarize(
        _cube_position_coordinates(empirical),
        _cube_position_coordinates(generated),
    )
    return model, {
        "emd_raw": raw["mean"],
        "emd_raw_std": raw["std"],
        "emd_raw_per_position": raw["per_position"],
        "emd_raw_per_source": raw["per_source"],
        "emd_cube_position": cube_position["mean"],
        "emd_cube_position_std": cube_position["std"],
        "emd_cube_position_per_position": cube_position["per_position"],
        "emd_cube_position_per_source": cube_position["per_source"],
        "empirical_terminal_success_mass": empirical_completion_mass,
        "generated_terminal_success_mass": generated_completion_mass,
        "terminal_success_mass_abs_error": abs(
            generated_completion_mass - empirical_completion_mass
        ),
        "cube_position_tolerance": cube_position_tolerance,
        "num_source_states": num_source_states,
        "num_samples_per_source_state": num_samples_per_source_state,
        "source_mode": source_mode,
        "backend": "ott_per_source_chunk",
    }


def _get_model_reconstructions_vae(
    model,
    source_observations: np.ndarray,
    num_samples_per_state: int,
):
    """Gets model samples from a VAE."""

    # NOTE: this method can be `jax.jit` if we return jax arrays instead of numpy arrays in sampling.
    # source observations shape: [B, observation_dim]
    # source actions shape: [B, action_dim]
    B = source_observations.shape[0]
    obs = np.expand_dims(source_observations, 1)  # (B,1,...)
    obs = np.tile(
        obs,
        (1, num_samples_per_state) + (1,) * (obs.ndim - 2),
    )  # (B,N,...)
    obs = obs.reshape((B * num_samples_per_state,) + obs.shape[2:])  # (B*N,...)

    # sample from model -> shape: [B * N, observation_dim]
    sample_fn = supply_sample_rng(
        model.encode_decode, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))
    )
    model_samples = sample_fn(obs)
    return model_samples


def gen_augment(
    batch,
    dataset,
    n_augment: int,
    gen_model,
    gen_model_config,
    generated_goal_transform=None,
    gen_stat_dataset=None,
):
    """Replace selected actor goals with futures generated from their state/actions."""
    batch = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if hasattr(x, "shape") else x, batch
    )
    batch_size = batch["observations"].shape[0]
    gen_stat_dataset = dataset if gen_stat_dataset is None else gen_stat_dataset
    if hasattr(gen_model, "rng"):
        rng, index_rng = jax.random.split(gen_model.rng)
        gen_model = gen_model.replace(rng=rng)
        gen_indices = jax.random.permutation(index_rng, batch_size)[:n_augment]
    else:
        gen_indices = np.random.choice(batch_size, size=n_augment, replace=False)
    gen_indices = jnp.asarray(gen_indices)


    # Record selected rows for augmentation diagnostics.
    aug_mask = jnp.zeros(batch_size, dtype=bool)
    aug_mask = aug_mask.at[gen_indices].set(True)
    batch["aug_mask"] = aug_mask

    # sample from the generative model
    source_observations = batch["observations"][gen_indices]
    source_actions = batch["actions"][gen_indices]

    if gen_model_config["normalize_observations"]:
        source_observations = normalize(
            source_observations,
            gen_stat_dataset.observation_statistics,
            gen_stat_dataset.norm_method,
        )
    gen_model, model_samples = gen_model.sample_jnp(source_observations, source_actions)
    real_goals = batch["actor_goals"][gen_indices]
    real_norm = jnp.linalg.norm(real_goals.reshape((real_goals.shape[0], -1)), axis=-1)
    model_norm = jnp.linalg.norm(
        model_samples.reshape((model_samples.shape[0], -1)), axis=-1
    )
    batch["aug_real_goal_norm"] = jnp.mean(real_norm)
    batch["aug_model_goal_norm"] = jnp.mean(model_norm)
    batch["aug_goal_norm_ratio"] = batch["aug_model_goal_norm"] / (
        batch["aug_real_goal_norm"] + 1e-8
    )


    # unnormalize if needed
    if gen_model_config["normalize_observations"]:
        model_samples = unnormalize(
            model_samples,
            gen_stat_dataset.observation_statistics,
            gen_stat_dataset.norm_method,
        )
    if generated_goal_transform is not None:
        model_samples = generated_goal_transform(model_samples)
        transformed_norm = jnp.linalg.norm(
            model_samples.reshape((model_samples.shape[0], -1)), axis=-1
        )
        batch["aug_model_goal_norm_transformed"] = jnp.mean(transformed_norm)
        batch["aug_goal_norm_transformed_ratio"] = batch[
            "aug_model_goal_norm_transformed"
        ] / (batch["aug_real_goal_norm"] + 1e-8)
    batch["actor_goals"] = _set_array_indices(
        batch["actor_goals"], gen_indices, model_samples
    )
    return batch, gen_model


def plot_maze_on_ax(maze_map, maze_unit, offset_x, offset_y, ax=None):
    """
    Draw the maze (0/1 grid) into world (x, y) coordinates on a given axis.

    maze_map : 2D array (H x W) of 0 (free) / 1 (wall)
    maze_unit, offset_x, offset_y : taken from env
    ax : matplotlib axis, if None a new figure/axis is created
    """
    maze_array = np.array(maze_map)
    H, W = maze_array.shape

    # Compute world-coordinate extent of the maze
    xmin = (-0.5 * maze_unit) - offset_x
    xmax = ((W - 0.5) * maze_unit) - offset_x
    ymin = (-0.5 * maze_unit) - offset_y
    ymax = ((H - 0.5) * maze_unit) - offset_y

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    cmap = plt.cm.colors.ListedColormap(["lightgray", "white"])
    ax.imshow(
        maze_array,
        cmap=cmap,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        interpolation="nearest",
    )

    # Nice styling
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax




def _render_cube_observation(
    env, observation, camera="front"
):
    """Restore a compact CubeEnv state observation and render it."""
    observation = np.asarray(observation, dtype=np.float64)
    expected_dim = 19 + 9 * int(env._num_cubes)
    if observation.shape != (expected_dim,):
        raise ValueError(
            f"Expected cube observation shape {(expected_dim,)}, "
            f"got {observation.shape}."
        )

    qpos = np.asarray(env._data.qpos).copy()
    qvel = np.zeros_like(env._data.qvel)
    qpos[env._arm_joint_ids] = observation[:6]
    qvel[env._arm_joint_ids] = observation[6:12]
    qpos[env._gripper_opening_joint_id] = (
        np.clip(observation[17] / 3.0, 0.0, 1.0) * 0.8
    )
    env.set_state(qpos, qvel)

    xyz_center = np.array([0.425, 0.0, 0.0])
    for cube_idx in range(env._num_cubes):
        start = 19 + 9 * cube_idx
        cube_joint = env._data.joint(f"object_joint_{cube_idx}")
        cube_joint.qpos[:3] = observation[start : start + 3] / 10.0 + xyz_center
        quat = observation[start + 3 : start + 7]
        quat_norm = np.linalg.norm(quat)
        cube_joint.qpos[3:] = (
            quat / quat_norm
            if quat_norm > 1e-8
            else np.array([1.0, 0.0, 0.0, 0.0])
        )

    for target_geom_ids in env._cube_target_geom_ids_list:
        for geom_id in target_geom_ids:
            env._model.geom(geom_id).rgba[3] = 0.0
    mujoco.mj_forward(env._model, env._data)
    return np.asarray(env.render(camera=camera), dtype=np.uint8)


def _render_puzzle_observation(
    env, observation, camera="front"
):
    """Restore a compact PuzzleEnv state observation and render it."""
    observation = np.asarray(observation, dtype=np.float64)
    num_buttons = int(env._num_buttons)
    expected_dim = 19 + 4 * num_buttons
    if observation.shape != (expected_dim,):
        raise ValueError(
            f"Expected puzzle observation shape {(expected_dim,)}, "
            f"got {observation.shape}."
        )

    qpos = np.asarray(env._data.qpos).copy()
    qvel = np.zeros_like(env._data.qvel)
    qpos[env._arm_joint_ids] = observation[:6]
    qvel[env._arm_joint_ids] = observation[6:12]
    qpos[env._gripper_opening_joint_id] = (
        np.clip(observation[17] / 3.0, 0.0, 1.0) * 0.8
    )

    button_states = np.empty(num_buttons, dtype=np.int32)
    for button_idx in range(num_buttons):
        start = 19 + 4 * button_idx
        button_states[button_idx] = int(np.argmax(observation[start : start + 2]))
        button_joint = env._model.joint(f"buttonbox_joint_{button_idx}")
        qpos[button_joint.qposadr[0]] = observation[start + 2] / 120.0
        qvel[button_joint.dofadr[0]] = observation[start + 3]

    env.set_state(qpos, qvel, button_states)
    return np.asarray(env.render(camera=camera), dtype=np.uint8)


def _plot_manipulation_futures(
    model, dataset, config, env, source_idxs, render_observation, title,
    num_samples_per_state, stat_dataset=None, render_dataset=None, camera="front",
):
    """Render one source, empirical mean, and generated mean per row."""
    if num_samples_per_state < 1 or len(source_idxs) < 1:
        raise ValueError("Plot source and sample counts must be positive.")
    stat_dataset = dataset if stat_dataset is None else stat_dataset
    render_dataset = dataset if render_dataset is None else render_dataset
    count = len(source_idxs)
    source_batch = dataset.sample(count, idxs=source_idxs, evaluation=True)
    empirical_batch = render_dataset.sample(
        count * num_samples_per_state,
        idxs=np.repeat(source_idxs, num_samples_per_state),
        evaluation=True,
    )
    empirical = np.asarray(empirical_batch["future_observations"]).reshape(
        count, num_samples_per_state, -1
    )
    observations = np.asarray(source_batch["observations"])
    actions = np.asarray(source_batch.get("condition_actions", source_batch["actions"]))
    if config["normalize_observations"]:
        observations = normalize(
            observations, stat_dataset.observation_statistics, stat_dataset.norm_method
        )
    model, generated = model.sample(
        np.repeat(observations, num_samples_per_state, axis=0),
        np.repeat(actions, num_samples_per_state, axis=0),
    )
    if config["normalize_observations"]:
        generated = unnormalize(
            generated, stat_dataset.observation_statistics, stat_dataset.norm_method
        )
    generated = np.asarray(generated, dtype=np.float32).reshape(count, num_samples_per_state, -1)
    sources = np.asarray(render_dataset.dataset["observations"])[source_idxs]

    def mean_render(states):
        images = np.stack([render_observation(env, state, camera) for state in states])
        return np.clip(images.mean(axis=0), 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(count, 3, figsize=(7.35, 2.65 * count), squeeze=False)
    for row in range(count):
        panels = (
            render_observation(env, sources[row], camera),
            mean_render(empirical[row]),
            mean_render(generated[row]),
        )
        for column, (ax, panel) in enumerate(zip(axes[row], panels)):
            ax.imshow(panel)
            ax.axis("off")
            if row == 0:
                ax.set_title(("Source", "Empirical", "Generated")[column], fontsize=11)
    fig.suptitle(
        f"{title}: action chunk {config['action_chunk_size']}, "
        f"discount {dataset.config['discount']:g}, {num_samples_per_state} samples",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.4)
    return model, {"state": fig}


def plot_chunk_samples_cube(
    model, dataset, config, env, num_samples_per_state=64, num_plot_states=3,
    stat_dataset=None, render_dataset=None, camera="front", source_mode="random",
):
    """Compare empirical and generated cube future states."""
    candidates = _cube_source_candidate_idxs(dataset, config, source_mode)
    source_idxs = np.random.choice(candidates, size=min(num_plot_states, len(candidates)), replace=False)
    return _plot_manipulation_futures(
        model, dataset, config, env, source_idxs, _render_cube_observation, "Cube futures",
        num_samples_per_state, stat_dataset, render_dataset, camera,
    )


def plot_chunk_samples_puzzle(
    model, dataset, config, env, num_samples_per_state=64, num_plot_states=3,
    stat_dataset=None, render_dataset=None, camera="front",
):
    """Compare empirical and generated puzzle future states."""
    source_idxs = dataset.get_random_idxs(min(num_plot_states, len(dataset.dataset["observations"])))
    return _plot_manipulation_futures(
        model, dataset, config, env, source_idxs, _render_puzzle_observation, "Puzzle futures",
        num_samples_per_state, stat_dataset, render_dataset, camera,
    )


def plot_samples_in_maze(
    env,
    model_samples,
    plot_source_observations,
    plot_source_actions,
    num_plot_states,
    num_samples_per_state,
    plot_source_trajectories,
):
    """
    env: a MazeEnv or BallEnv instance
    model_samples: [num_plot_states * num_samples_per_state, obs_dim]
    plot_source_observations: [num_plot_states, obs_dim]
    plot_source_actions: [num_plot_states, action_dim] or None
    """
    if plot_source_actions is not None:
        # action dimension should be 2 for plotting
        assert plot_source_actions.shape[-1] == 2
    # reshape model samples to [num_plot_states, num_samples_per_state, observation_dim]
    model_samples = model_samples.reshape(
        (num_plot_states, num_samples_per_state, model_samples.shape[-1])
    )
    # model_samples = model_samples.reshape(
    #     (num_plot_states, num_samples_per_state) + model_samples.shape[1:]
    # )

    fig, axs = plt.subplots(1, num_plot_states, figsize=(5 * num_plot_states, 5))
    if num_plot_states == 1:
        axs = [axs]

    for i in range(num_plot_states):
        ax = axs[i]

        # 1) Draw maze background in world coords
        plot_maze_on_ax(
            maze_map=env.maze_map,
            maze_unit=env._maze_unit,
            offset_x=env._offset_x,
            offset_y=env._offset_y,
            ax=ax,
        )

        # 2) Overlay model samples (assumes first two dims are x, y in env coords)
        ax.scatter(
            model_samples[i, :, 0],
            model_samples[i, :, 1],
            alpha=0.1,
            label="Model Samples",
        )
        # overlay source trajectory
        if plot_source_trajectories is not None:
            traj = plot_source_trajectories[i]
            ax.scatter(
                traj[:, 0],
                traj[:, 1],
                color="orange",
                # linewidth=2,
                label="Source Trajectory",
                # alpha=0.8,
                alpha=0.1,
            )

        # 3) Overlay source state
        if plot_source_observations is not None:
            ax.scatter(
                plot_source_observations[i, 0],
                plot_source_observations[i, 1],
                color="red",
                label="Source State",
            )
        # plot action vector if not none on top of source state
        if plot_source_actions is not None:
            ax.quiver(
                plot_source_observations[i, 0],
                plot_source_observations[i, 1],
                plot_source_actions[i, 0],
                plot_source_actions[i, 1],
                angles="xy",
                scale_units="xy",
                scale=0.6,
                color="green",
                # label='Action Vector',
            )

        ax.set_title(f"Source State {i}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
    fig.savefig("test.png")
    return fig




def plot_samples_xy(
    model,
    dataset,
    config,
    env,
    num_samples_per_state: int = 256,
    num_plot_states: int = 5,
    num_future_states: int = 5,
    plot_actions: bool = False,
    plot_images: bool = False,
    plot_maze: bool = True,
    pretrained_encoder: Any = None,
    pretrained_decoder: Any = None,
    stat_dataset = None,
):
    """Plots samples in xy space for a given model and dataset."""

    assert hasattr(model, "sample")

    normalize_observations = config["normalize_observations"]
    # first get trajectory starts and ends -> episode is [start, end)
    stat_dataset = dataset if stat_dataset is None else stat_dataset
    observations = dataset.dataset["observations"]
    episode_starts = dataset.initial_locs
    episode_ends = dataset.terminal_locs
    lengths = episode_ends - episode_starts + 1
    # get starts and ends with length > 1
    valid_indices = np.where(lengths > 1)[0]
    episode_starts = episode_starts[valid_indices]
    episode_ends = episode_ends[valid_indices]
    num_trajs = len(episode_starts)

    def _sample_and_plot(start=True):
        nonlocal model
        # select a few source states to plot
        plot_indices = np.random.choice(num_trajs, size=num_plot_states, replace=False)
        if start:
            starts = episode_starts[plot_indices]
        else:
            # sample random indices from start to end uniformly
            starts = np.array(
                [
                    np.random.randint(episode_starts[i], episode_ends[i] + 1)
                    for i in plot_indices
                ]
            )
        ends = episode_ends[plot_indices]
        plot_source_observations = np.stack(
            [observations[starts[i]] for i in range(len(starts))]
        )
        plot_source_trajectories = [
            observations[starts[i] : ends[i] + 1] for i in range(len(starts))
        ]
        plot_source_actions = np.asarray(dataset.dataset["actions"][starts])
        model_source_actions = _source_actions_for_model(dataset, starts, config)
        raw_source_obs = plot_source_observations

        # normalize if needed
        if normalize_observations:
            plot_source_observations = normalize(
                plot_source_observations,
                stat_dataset.observation_statistics,
                stat_dataset.norm_method,
            )
        if pretrained_encoder is not None:
            # encode source observations
            plot_source_observations = pretrained_encoder(plot_source_observations)

        # get model samples
        model, model_samples = sample_model_states(
            model,
            plot_source_observations,
            model_source_actions,
            num_samples_per_state,
        )


        # unnormalize for plotting
        if normalize_observations:
            plot_source_observations = unnormalize(
                plot_source_observations,
                stat_dataset.observation_statistics,
                stat_dataset.norm_method,
            )
            model_samples = unnormalize(
                model_samples, stat_dataset.observation_statistics, stat_dataset.norm_method
            )
        if pretrained_decoder is not None and hasattr(pretrained_decoder, "decode"):
            # decode model samples
            model_samples = pretrained_decoder.decode(model_samples)
            if not pretrained_decoder.config["scale_out"]:
                # scale back to [0, 255]
                model_samples = (model_samples * 255.0).clip(0, 255)

        raw_samples = model_samples
        raw_source_obs_imgs = raw_source_obs
        # reshape model samples to [num_plot_states, num_samples_per_state, observation_dim]
        model_samples = model_samples.reshape(
            (num_plot_states, num_samples_per_state) + model_samples.shape[1:]
        )
        if not plot_actions:
            plot_source_actions = None
        # raw_samples = model_samples
        raw_samples = raw_samples.reshape(
            (num_plot_states, num_samples_per_state) + raw_samples.shape[1:]
        )
        raw_trajectories = plot_source_trajectories
        # also obtain source trajectories
        fig = None
        if plot_maze:
            fig = plot_samples_in_maze(
                env,
                model_samples=model_samples,
                plot_source_observations=plot_source_observations,
                plot_source_actions=plot_source_actions,
                num_plot_states=num_plot_states,
                num_samples_per_state=num_samples_per_state,
                plot_source_trajectories=plot_source_trajectories,
            )
        img_fig = None
        gt_img_fig = None
        can_plot_images = (
            plot_images
            and raw_samples.ndim >= 5
            and raw_source_obs_imgs.ndim >= 4
        )
        if plot_images and not can_plot_images:
            print(
                "[plot_samples_xy] Skipping image grids because samples are not images: "
                f"raw_samples_shape={raw_samples.shape}, "
                f"raw_source_obs_shape={raw_source_obs_imgs.shape}"
            )
        if can_plot_images:
            #  plot_source observations is 5 x 64 x 64 x 3
            # and then model samples is like 5 x 32 x 64 x 64 x 3
            # we want a big grid with source observation as first column, then samples next to it in rows
            # e.g. one huge image that is 5 x (1 + 32) x 64 x 64 x 3
            num_samples_image = num_samples_per_state
            image_grid = []
            for i in range(num_plot_states):
                row_images = []
                # add source observation first
                row_images.append(raw_source_obs_imgs[i])
                # then add samples
                for j in range(num_samples_image):
                    row_images.append(raw_samples[i, j])
                # concatenate row images horizontally
                row_image = np.concatenate(row_images, axis=1)
                image_grid.append(row_image)
            # concatenate all rows vertically
            final_image = np.concatenate(image_grid, axis=0)
            # # plot final image
            # fig2, ax2 = plt.subplots(figsize=(10, 10))
            # img_fig = fig2
            # ax2.imshow(final_image.astype(np.uint8))
            # ax2.axis("off")
            # fig2.savefig("test_images.png")
            # plt.close(fig2)
            img_fig = final_image

            # also create img using raw_trajectoreis, i.e. this is ground truth trajectories
            gt_image_grid = []
            # plot up to num_samples_per_state frames from each trajectory, also shuffle frames w/in a trajectory to not be in order
            # we will pad w/ blank images if trajectory is too short
            for i in range(num_plot_states):
                row_images = []
                traj = raw_trajectories[i]
                traj_length = traj.shape[0]
                # get indices to sample from trajectory
                if traj_length >= num_samples_image:
                    selected_indices = np.random.choice(
                        traj_length, size=num_samples_image, replace=False
                    )
                else:
                    selected_indices = np.arange(traj_length)
                    # shuffle selected indices
                    np.random.shuffle(selected_indices)
                # add images to row
                # for now encode, potentially store images in dataset for viz

                traj_row = traj[selected_indices]
                traj_row = np.asarray(traj_row)
                for idx in range(len(traj_row)):
                    row_images.append(traj_row[idx])
                # pad w/ blank images if needed
                pad_like = traj_row[0] if len(traj_row) > 0 else None
                while len(row_images) < num_samples_image:
                    if pad_like is None:
                        pad_like = traj[0]
                    row_images.append(np.zeros_like(pad_like))
                # concatenate row images horizontally
                row_image = np.concatenate(row_images, axis=1)
                gt_image_grid.append(row_image)
            # concatenate all rows vertically
            final_gt_image = np.concatenate(gt_image_grid, axis=0)
            # plot final image
            # fig3, ax3 = plt.subplots(figsize=(10, 10))
            # ax3.imshow(final_gt_image.astype(np.uint8))
            # ax3.axis("off")
            # fig3.savefig("test_gt_images.png")
            # plt.close(fig3)
            gt_img_fig = final_gt_image

        return fig, img_fig, gt_img_fig

    # figs = {"start": _sample_and_plot(), "uniform": _sample_and_plot(start=False)}
    figs = {}
    fig, img_fig, gt_img_fig = _sample_and_plot(start=True)
    figs["start"] = fig
    figs["start_img"] = img_fig
    figs["start_gt_img"] = gt_img_fig
    fig, img_fig, gt_img_fig = _sample_and_plot(start=False)
    figs["uniform"] = fig
    figs["uniform_img"] = img_fig
    figs["uniform_gt_img"] = gt_img_fig

    return model, figs


def plot_samples_xy_vae(
    model,
    dataset,
    config,
    env,
    num_samples_per_state: int = 256,
    num_plot_states: int = 5,
    plot_images: bool = False,
    debug=True,
):
    normalize_observations = config["normalize_observations"]
    """Plots samples in xy space for a given model and dataset. samples from VAE"""

    assert hasattr(model, "sample")

    # first get trajectory starts and ends -> episode is [start, end)
    observations = dataset.dataset["observations"]
    episode_starts = dataset.initial_locs
    episode_ends = dataset.terminal_locs
    lengths = episode_ends - episode_starts + 1
    # get starts and ends with length > 1
    valid_indices = np.where(lengths > 1)[0]
    episode_starts = episode_starts[valid_indices]
    episode_ends = episode_ends[valid_indices]
    num_trajs = len(episode_starts)

    def _sample_and_plot(start=True):
        nonlocal model
        # select a few source states to plot
        plot_indices = np.random.choice(num_trajs, size=num_plot_states, replace=False)
        if start:
            starts = episode_starts[plot_indices]
        else:
            # sample random indices from start to end uniformly
            starts = np.array(
                [
                    np.random.randint(episode_starts[i], episode_ends[i] + 1)
                    for i in plot_indices
                ]
            )
        ends = episode_ends[plot_indices]
        plot_source_observations = np.stack(
            [observations[starts[i]] for i in range(len(starts))]
        )
        plot_source_trajectories = [
            observations[starts[i] : ends[i] + 1] for i in range(len(starts))
        ]
        plot_source_actions = dataset.dataset["actions"][starts]

        # normalize if needed
        if normalize_observations:
            plot_source_observations = normalize(
                plot_source_observations,
                dataset.observation_statistics,
                dataset.norm_method,
            )

        # get model samples
        model_samples = _get_model_samples_vae(
            model,
            plot_source_observations,
            num_samples_per_state,
        )
        model_reconstructions = _get_model_reconstructions_vae(
            model,
            plot_source_observations,
            num_samples_per_state,
        )

        # unnormalize for plotting
        if normalize_observations:
            plot_source_observations = unnormalize(
                plot_source_observations,
                dataset.observation_statistics,
                dataset.norm_method,
            )
            model_samples = unnormalize(
                model_samples, dataset.observation_statistics, dataset.norm_method
            )
            model_reconstructions = unnormalize(
                model_reconstructions,
                dataset.observation_statistics,
                dataset.norm_method,
            )
        else:

            if not config["scale_out"]:
                # scale back to [0, 255]
                model_samples = (model_samples * 255.0).clip(0, 255)
                model_reconstructions = (model_reconstructions * 255.0).clip(0, 255)

        # reshape model samples to [num_plot_states, num_samples_per_state, observation_dim]
        model_samples = model_samples.reshape(
            (num_plot_states, num_samples_per_state) + model_samples.shape[1:]
        )
        model_reconstructions = model_reconstructions.reshape(
            (num_plot_states, num_samples_per_state) + model_reconstructions.shape[1:]
        )
        raw_samples = model_samples
        raw_reconstructions = model_reconstructions
        raw_source_obs = plot_source_observations
        raw_trajectories = plot_source_trajectories
        fig = None
        img_fig = None
        gt_img_fig = None
        if debug:
            import pdb

            pdb.set_trace()
        if plot_images:
            #  plot_source observations is 5 x 64 x 64 x 3
            # and then model samples is like 5 x 32 x 64 x 64 x 3
            # we want a big grid with source observation as first column, then samples next to it in rows
            # e.g. one huge image that is 5 x (1 + 32) x 64 x 64 x 3
            num_samples_image = num_samples_per_state
            image_grid = []
            for i in range(num_plot_states):
                row_images = []
                # add source observation first
                row_images.append(raw_source_obs[i])
                # then add samples
                for j in range(num_samples_image):
                    row_images.append(raw_reconstructions[i, j])
                # concatenate row images horizontally
                row_image = np.concatenate(row_images, axis=1)
                image_grid.append(row_image)
            # concatenate all rows vertically
            final_image = np.concatenate(image_grid, axis=0)
            img_fig = final_image

            # create grid of model samples
            image_grid = []
            for i in range(num_plot_states):
                row_images = []
                # add source observation first
                row_images.append(raw_source_obs[i])
                # then add samples
                for j in range(num_samples_image):
                    row_images.append(raw_samples[i, j])
                # concatenate row images horizontally
                row_image = np.concatenate(row_images, axis=1)
                image_grid.append(row_image)
            # concatenate all rows vertically
            final_image = np.concatenate(image_grid, axis=0)
            marginal_img_fig = final_image

        return fig, img_fig, marginal_img_fig

    # figs = {"start": _sample_and_plot(), "uniform": _sample_and_plot(start=False)}
    figs = {}
    fig, img_fig, marginal_img_fig = _sample_and_plot(start=False)
    figs["uniform"] = fig
    figs["uniform_img_vae"] = img_fig
    figs["marginal_img_vae"] = marginal_img_fig

    return figs


def compute_log_likelihood(
    model,
    dataset,
    config,
    num_samples_per_source_state: int = 256,
    num_eval_states: int = 50,
    discount: Optional[float] = None,
    pretrained_encoder: Any = None,
    stat_dataset = None,
):
    """Computes log likelihoods of future states conditioned on source states and policy."""

    assert hasattr(model, "compute_log_likelihood")

    discount = config["discount"] if discount is None else discount
    normalize_observations = config["normalize_observations"]
    stat_dataset = dataset if stat_dataset is None else stat_dataset
    # first get trajectory starts and ends -> episode is [start, end)
    observations = dataset.dataset["observations"]
    # episode_starts, episode_ends = dataset.trajectory_boundaries()
    episode_starts = dataset.initial_locs
    episode_ends = dataset.terminal_locs
    lengths = episode_ends - episode_starts + 1
    # get starts and ends with length > 1
    valid_indices = np.where(lengths > 1)[0]
    episode_starts = episode_starts[valid_indices]
    episode_ends = episode_ends[valid_indices]
    num_trajs = len(episode_starts)

    # ony use a subset of the trajectories for evaluation
    if num_eval_states < num_trajs:
        selected_indices = np.random.choice(
            num_trajs, size=num_eval_states, replace=False
        )
        episode_starts = episode_starts[selected_indices]
        episode_ends = episode_ends[selected_indices]
        num_trajs = len(episode_starts)

    # sample true model samples via geometric sampling
    # do this over every trajectory in the dataset, as those are the source states we care about
    future_samples = []
    for start, end in zip(episode_starts, episode_ends):
        traj_length = end - start

        # sample `num_samples_per_source_state` samples from the geometric horizon distribution
        indices = np.random.geometric(p=1 - discount, size=num_samples_per_source_state)

        # if the index is larger than the trajectory length, then we clip at the trajectory length itself.
        indices = np.where(indices > traj_length, traj_length, indices)

        # add the samples to the true samples
        assert np.all([start + index <= end for index in indices])
        future_traj_samples = np.stack(
            [observations[start + index] for index in indices]
        )
        future_samples.append(future_traj_samples)
    # size [num_trajs * num_samples_per_source_state, observation_dim]. this is not normalized.
    future_samples = np.concatenate(future_samples, axis=0)
    if normalize_observations:
        future_samples = normalize(
            future_samples, stat_dataset.observation_statistics, stat_dataset.norm_method
        )
    if pretrained_encoder is not None:
        # encode future samples
        future_samples = pretrained_encoder(future_samples)
    # now we compute model log likelihood conditioned on the source states (all unnormalized right now).
    source_observations = np.stack([observations[start] for start in episode_starts])
    source_observations = np.expand_dims(source_observations, 1)
    source_observations = np.tile(
        source_observations,
        (1, num_samples_per_source_state) + (1,) * (source_observations.ndim - 2),
    )  # (num_trajs, N, ...)

    # flatten only traj and N dims, keep the rest
    source_observations = source_observations.reshape(
        (num_trajs * num_samples_per_source_state,) + source_observations.shape[2:]
    )

    source_actions = dataset.dataset["actions"][episode_starts]
    source_actions = np.expand_dims(source_actions, 1)
    source_actions = np.tile(source_actions, (1, num_samples_per_source_state, 1))
    source_actions = np.reshape(source_actions, (-1, source_actions.shape[-1]))

    # normalize now.
    if normalize_observations:
        source_observations = normalize(
            source_observations, stat_dataset.observation_statistics, stat_dataset.norm_method
        )
    if pretrained_encoder is not None:
        # encode source observations
        source_observations = pretrained_encoder(source_observations)

    # now get the model log likelihood of future samples from source samples. [B * N] -> [B, N]
    # if we wanted to normalize, all samples should be normalized now.
    model, log_likelihoods = model.compute_log_likelihood(
        future_samples, source_observations, source_actions
    )
    log_likelihoods = log_likelihoods.reshape((num_trajs, num_samples_per_source_state))

    # take the mean of the log likelihoods across both batch and num samples per source state.
    return model, np.mean(log_likelihoods)
