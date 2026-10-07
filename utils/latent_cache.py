"""Utilities for replacing a dataset's image table with frozen encoder latents."""

import jax
import numpy as np
import tqdm

from utils.datasets import Dataset


def encode_observation_table(observations, encoder, batch_size):
    """Encode a full observation table in batches and return a host NumPy array."""
    if batch_size <= 0:
        raise ValueError(f"latent_cache_batch_size must be positive, got {batch_size}.")
    if not isinstance(observations, np.ndarray):
        raise TypeError("Latent caching currently requires array observations.")
    if observations.ndim < 2:
        raise ValueError(
            f"Expected a batched observation table, got shape {observations.shape}."
        )

    encoded_batches = []
    for start in tqdm.trange(
        0,
        len(observations),
        batch_size,
        smoothing=0.1,
        dynamic_ncols=True,
        desc="Caching encoder latents",
    ):
        encoded = encoder(observations[start : start + batch_size])
        encoded_batches.append(np.asarray(jax.device_get(encoded), dtype=np.float32))

    if not encoded_batches:
        raise ValueError("Cannot cache latents for an empty dataset.")
    return np.concatenate(encoded_batches, axis=0)


def cache_dataset_observations(dataset, encoder, batch_size):
    """Return a Dataset whose observation table is replaced by encoder latents.

    Other arrays are shared with the source dataset. Derived next states, goals,
    histories, and chunks therefore remain index-based views of one latent table.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError(
            "cache_dataset_observations expects the base utils.datasets.Dataset."
        )

    latent_observations = encode_observation_table(
        np.asarray(dataset["observations"]),
        encoder,
        batch_size=batch_size,
    )
    fields = {key: dataset[key] for key in dataset.keys()}
    fields["observations"] = latent_observations
    return Dataset.create(**fields)
