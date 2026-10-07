"""Online frozen-prior bootstrap action sampling."""

import jax
import jax.numpy as jnp


@jax.jit
def _sample_action_prior_batched(action_prior, observations, seeds):
    return jax.vmap(
        lambda observation, seed: action_prior.sample_actions(
            observations=observation,
            goals=None,
            seed=seed,
            temperature=1.0,
        )
    )(observations, seeds)


def inject_online_action_prior(
    batch, action_prior, rng, *, use_future_bootstrap, state_mchc=False
):
    """Sample frozen-prior actions at TD2 next states or MCHC future states."""
    batch = dict(batch)
    if state_mchc:
        if not use_future_bootstrap:
            raise ValueError("TDFlow online prior currently supports MC_HC_TD2 only")
        # Query the raw state at the exact future index used by the MCHC teacher.
        obs_key, action_key = "mc_offset_value_goals", "mc_offset_value_goal_actions"
        observations = jnp.asarray(batch[obs_key])
        rng, sample_rng = jax.random.split(rng)
        sampled = _sample_action_prior_batched(
            action_prior, observations, jax.random.split(sample_rng, len(observations))
        )
        if sampled.shape != batch[action_key].shape:
            raise ValueError(f"Prior shape {sampled.shape} != {action_key} shape {batch[action_key].shape}")
        batch[action_key] = sampled
        return batch, rng
    if use_future_bootstrap:
        mask_key, observation_key, action_key, valid_key = (
            "future_bootstrap_uses_action_prior", "future_observations",
            "future_chunk_actions", "future_chunk_action_valids",
        )
    else:
        mask_key, observation_key, action_key, valid_key = (
            "bootstrap_uses_action_prior", "bootstrap_observations",
            "bootstrap_actions", "bootstrap_action_valids",
        )
    if mask_key not in batch:
        return batch, rng
    observations = jnp.asarray(batch[observation_key])
    rng, sample_rng = jax.random.split(rng)
    seeds = jax.random.split(sample_rng, len(observations))
    sampled_actions = _sample_action_prior_batched(
        action_prior, observations, seeds
    )
    recorded_actions = jnp.asarray(batch[action_key])
    # H1 actors omit the singleton chunk axis used by GHM datasets.
    if (
        sampled_actions.ndim == 2
        and recorded_actions.ndim == 3
        and recorded_actions.shape[1] == 1
    ):
        sampled_actions = sampled_actions[:, None, :]
    if sampled_actions.shape != recorded_actions.shape:
        raise ValueError(
            f"Prior action shape {sampled_actions.shape} does not match "
            f"{action_key} shape {recorded_actions.shape}"
        )
    mask = jnp.asarray(batch[mask_key], dtype=bool)
    action_mask = mask.reshape((len(mask),) + (1,) * (sampled_actions.ndim - 1))
    batch[action_key] = jnp.where(
        action_mask, sampled_actions, recorded_actions
    )
    if valid_key in batch:
        valid_mask = mask.reshape(
            (len(mask),) + (1,) * (jnp.asarray(batch[valid_key]).ndim - 1)
        )
        batch[valid_key] = jnp.where(
            valid_mask,
            jnp.ones_like(batch[valid_key]),
            jnp.asarray(batch[valid_key]),
        )
    return batch, rng
