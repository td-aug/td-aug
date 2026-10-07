"""Goal-representation helpers shared by datasets and evaluation."""

import jax.numpy as jnp
import numpy as np


def _locomaze_oracle_representation(
    observations, array_module, single_observation_dim=None
):
    """Extract the agent XY position from AntMaze/HumanoidMaze states."""
    observations = array_module.asarray(observations)
    if observations.shape[-1] < 2:
        raise ValueError(
            "Locomaze observations must contain at least XY coordinates, "
            f"got dimension {observations.shape[-1]}."
        )
    if single_observation_dim is not None:
        if single_observation_dim < 2:
            raise ValueError("single_observation_dim must be at least 2.")
        if observations.shape[-1] % single_observation_dim != 0:
            raise ValueError(
                "Locomaze goal dimension must be a multiple of "
                f"{single_observation_dim}, got {observations.shape[-1]}."
            )
        observations = observations.reshape(
            *observations.shape[:-1], -1, single_observation_dim
        )[..., -1, :]
    return observations[..., :2].astype(array_module.float32)


def locomaze_oracle_representation(observations, single_observation_dim=None):
    """Return the OGBench AntMaze/HumanoidMaze oracle XY representation."""
    return _locomaze_oracle_representation(
        observations, np, single_observation_dim=single_observation_dim
    )


def locomaze_oracle_representation_jax(
    observations, single_observation_dim=None
):
    """JAX implementation of the locomaze oracle XY representation."""
    return _locomaze_oracle_representation(
        observations, jnp, single_observation_dim=single_observation_dim
    )


def _cube_oracle_representation(observations, num_cubes: int, array_module):
    """Extract normalized cube XYZ positions from Cube observations.

    Cube observations contain 19 proprioceptive values followed by nine values
    per cube: XYZ, quaternion, cos(yaw), and sin(yaw). Frame-stacked goals are
    reduced to their final frame before extracting cube positions.
    """
    observations = array_module.asarray(observations)
    single_observation_dim = 19 + 9 * num_cubes
    if observations.shape[-1] % single_observation_dim != 0:
        raise ValueError(
            "Cube goal dimension must be a multiple of "
            f"{single_observation_dim}, got {observations.shape[-1]}."
        )

    observations = observations.reshape(
        *observations.shape[:-1], -1, single_observation_dim
    )[..., -1, :]
    cube_reps = []
    for cube_idx in range(num_cubes):
        start = 19 + cube_idx * 9
        cube_reps.append(observations[..., start : start + 3])
    return array_module.concatenate(cube_reps, axis=-1).astype(array_module.float32)


def cube_oracle_representation(observations, num_cubes: int):
    """NumPy implementation used while preparing offline datasets."""
    return _cube_oracle_representation(observations, num_cubes, np)


def cube_oracle_representation_jax(observations, num_cubes: int):
    """JAX implementation used to project generated states during training."""
    return _cube_oracle_representation(observations, num_cubes, jnp)


def _puzzle_oracle_representation(observations, num_buttons: int, array_module):
    """Extract binary button states from full Puzzle observations."""
    observations = array_module.asarray(observations)
    single_observation_dim = 19 + 4 * num_buttons
    if observations.shape[-1] % single_observation_dim != 0:
        raise ValueError(
            "Puzzle goal dimension must be a multiple of "
            f"{single_observation_dim}, got {observations.shape[-1]}."
        )

    observations = observations.reshape(
        *observations.shape[:-1], -1, single_observation_dim
    )[..., -1, :]
    button_logits = array_module.stack(
        [
            observations[..., 19 + 4 * button_idx : 21 + 4 * button_idx]
            for button_idx in range(num_buttons)
        ],
        axis=-2,
    )
    return array_module.argmax(button_logits, axis=-1).astype(array_module.float32)


def puzzle_oracle_representation(observations, num_buttons: int):
    """Return the Puzzle oracle representation used by OGBench oraclerep."""
    return _puzzle_oracle_representation(observations, num_buttons, np)


def puzzle_oracle_representation_jax(observations, num_buttons: int):
    """JAX implementation of the Puzzle oracle representation."""
    return _puzzle_oracle_representation(observations, num_buttons, jnp)


def _scene_oracle_representation(
    observations, num_cubes: int, num_buttons: int, array_module
):
    """Extract the OGBench Scene oracle state from full observations.

    Scene observations already contain normalized cube positions, one-hot
    button states, and scaled drawer/window positions. Frame-stacked goals are
    reduced to their final frame before extracting those fields.
    """
    observations = array_module.asarray(observations)
    single_observation_dim = 19 + 9 * num_cubes + 4 * num_buttons + 4
    if observations.shape[-1] % single_observation_dim != 0:
        raise ValueError(
            "Scene goal dimension must be a multiple of "
            f"{single_observation_dim}, got {observations.shape[-1]}."
        )

    observations = observations.reshape(
        *observations.shape[:-1], -1, single_observation_dim
    )[..., -1, :]
    cube_reps = [
        observations[..., 19 + 9 * cube_idx : 22 + 9 * cube_idx]
        for cube_idx in range(num_cubes)
    ]
    button_start = 19 + 9 * num_cubes
    button_logits = array_module.stack(
        [
            observations[
                ..., button_start + 4 * button_idx : button_start + 4 * button_idx + 2
            ]
            for button_idx in range(num_buttons)
        ],
        axis=-2,
    )
    button_reps = array_module.argmax(button_logits, axis=-1).astype(
        array_module.float32
    )
    fixture_start = button_start + 4 * num_buttons
    drawer_rep = observations[..., fixture_start : fixture_start + 1]
    window_rep = observations[..., fixture_start + 2 : fixture_start + 3]
    return array_module.concatenate(
        [*cube_reps, button_reps, drawer_rep, window_rep], axis=-1
    ).astype(array_module.float32)


def scene_oracle_representation(observations, num_cubes: int, num_buttons: int):
    """Return the Scene oracle representation used by OGBench oraclerep."""
    return _scene_oracle_representation(observations, num_cubes, num_buttons, np)


def scene_oracle_representation_jax(
    observations, num_cubes: int, num_buttons: int
):
    """JAX implementation of the Scene oracle representation."""
    return _scene_oracle_representation(observations, num_cubes, num_buttons, jnp)


def _copy_dataset_with(dataset, **updates):
    """Copy a Dataset or plain data dictionary and apply field updates."""
    if isinstance(dataset, dict):
        result = dataset.copy()
        result.update(updates)
        return result
    return dataset.copy(add_or_replace=updates)


def add_cube_oracle_representations(dataset, num_cubes: int):
    """Return a dataset copy containing oracle representations for every state."""
    oracle_reps = cube_oracle_representation(dataset["observations"], num_cubes)
    return _copy_dataset_with(dataset, oracle_reps=oracle_reps)


def add_locomaze_oracle_representations(dataset):
    """Return a dataset copy containing agent XY for every locomaze state."""
    oracle_reps = locomaze_oracle_representation(dataset["observations"])
    return _copy_dataset_with(dataset, oracle_reps=oracle_reps)


def add_puzzle_oracle_representations(dataset, num_buttons: int):
    """Return a dataset copy containing binary Puzzle button-state goals."""
    oracle_reps = puzzle_oracle_representation(
        dataset["observations"], num_buttons
    )
    return _copy_dataset_with(dataset, oracle_reps=oracle_reps)


def add_scene_oracle_representations(
    dataset, num_cubes: int, num_buttons: int
):
    """Return a dataset copy containing Scene oracle states for every step."""
    oracle_reps = scene_oracle_representation(
        dataset["observations"], num_cubes, num_buttons
    )
    return _copy_dataset_with(dataset, oracle_reps=oracle_reps)
