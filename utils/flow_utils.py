import math
from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Euler, Midpoint, ODETerm, diffeqsolve

from utils.types import ApplyFn, Params, PRNGKey


def validate_flow_config(config):
    """Validate model inputs and active training settings."""
    if config.get("state_only", False):
        raise ValueError("state_only=True is not supported; flows condition on state and action.")
    if config.get("n_step", None):
        raise ValueError("Legacy n_step frame stacking is not supported; use observation history.")
    if config.get("bootstrap_continuation_mode", "recorded") not in ("recorded", "policy_prior"):
        raise ValueError("Bootstrap continuation must use recorded actions or policy_prior.")


def _solver_from_name(solver_name: str):
    if solver_name == "midpoint":
        return Midpoint()
    if solver_name == "euler":
        return Euler()
    raise ValueError(f"Unknown flow sampler solver {solver_name!r}. Expected 'midpoint' or 'euler'.")


def sample_model_states(
    model, source_observations, source_actions, num_samples_per_state,
    sample_discount=None, sample_max_steps=None,
):
    """Sample futures in source-major order, preserving image and action-chunk axes."""
    observations = np.repeat(source_observations, num_samples_per_state, axis=0)
    actions = np.repeat(source_actions, num_samples_per_state, axis=0)
    if sample_discount is not None and hasattr(model, "sample_geometric"):
        return model.sample_geometric(
            observations, actions, discount=sample_discount, max_steps=sample_max_steps
        )
    return model.sample(observations, actions)


@partial(jax.jit, static_argnames=("apply_fn", "sampler_solver"))
def flow_sample(
    rng: PRNGKey,
    apply_fn: ApplyFn,
    params: Params,
    observation: np.ndarray,
    action: np.ndarray,
    sample_step_size: Union[float, np.ndarray],
    final_t: Union[float, np.ndarray] = 1.0,
    sampler_solver: str = "midpoint",
    sample_template: Optional[np.ndarray] = None,
) -> Tuple[jax.Array, PRNGKey]:
    """Samples from the probability path defined by `apply_fn({"params": params}).` This allows for arbitrary stopping in the middle (e.g. `0 < final_t < 1`)."""

    # sample initial x_0 ~ p_0
    rng, sample_key = jax.random.split(rng)
    sample_template = observation if sample_template is None else sample_template
    x_0 = jax.random.normal(sample_key, sample_template.shape)
    vmap = (
        (observation.ndim > 1)
        and (action.ndim > 1)
        and (observation.shape[0] == action.shape[0])
    )

    # set up variables to apply function
    variables = {"params": params}

    # define the ODE
    def _ode_fn(
        t: jax.Array,
        x_t: jax.Array,
        args: Tuple[jax.Array, jax.Array],
    ) -> jax.Array:

        # first convert t from float to jax array
        t = jnp.array(t)

        cond_observation, cond_action = args
        pred_v_t = apply_fn(variables, x_t, t, cond_observation, cond_action)

        return pred_v_t

    def _forward_fn(
        unbatched_x_0: jax.Array,
        unbatched_final_t: Union[float, jax.Array],
        unbatched_step_size: Union[float, jax.Array],
        unbatched_cond_observation: jax.Array,
        unbatched_cond_action: jax.Array,
    ) -> jax.Array:
        term = ODETerm(_ode_fn)
        solver = _solver_from_name(sampler_solver)

        solution = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=unbatched_final_t,
            dt0=unbatched_step_size,
            y0=unbatched_x_0,
            args=(unbatched_cond_observation, unbatched_cond_action),
        )
        (x_t,) = solution.ys

        return x_t

    if vmap:
        bs = observation.shape[0]

        # if `final_t` and `sample_step_size` are floats, then we need to make them have the same batch size for vmapping
        final_t = jnp.array(final_t)
        sample_step_size = jnp.array(sample_step_size)
        # Allow scalar/batched mixtures (e.g. final_t=1.0 with per-sample step sizes).
        if final_t.ndim == 0:
            final_t = jnp.tile(final_t, (bs,))
        if sample_step_size.ndim == 0:
            sample_step_size = jnp.tile(sample_step_size, (bs,))

        assert final_t.shape[0] == bs
        assert sample_step_size.shape[0] == bs

        _forward_fn = jax.vmap(_forward_fn, in_axes=(0, 0, 0, 0, 0))

    x_t = _forward_fn(x_0, final_t, sample_step_size, observation, action)
    return x_t, rng




@partial(
    jax.jit,
    static_argnames=(
        "apply_fn",
        "step_size",
        "trace_estimator",
        "num_hutchinson_samples",
    ),
)
def flow_log_likelihood(
    rng: PRNGKey,
    apply_fn: ApplyFn,
    params: Params,
    observation: np.ndarray,
    cond_observation: np.ndarray,
    cond_action: np.ndarray,
    step_size: float,
    trace_estimator: str = "hutchinson",
    num_hutchinson_samples: int = 1,
) -> jax.Array:
    """Compute the log probability of a given observation `observation` conditioned on `cond_observation, cond_action` using midpoint method. Can be batched."""

    vmap = observation.ndim > 1

    # set up variables to apply function
    variables = {"params": params}

    # we need to define the div operator
    def _ode_fn(
        x_t: jax.Array,
        t: jax.Array,
        cond_observation: jax.Array,
        cond_action: jax.Array,
    ) -> jax.Array:

        # first convert t from float to jax array
        t = jnp.array(t)

        # compute velocity via forward pass of model
        pred_v_t = apply_fn(variables, x_t, t, cond_observation, cond_action)

        return pred_v_t

    if num_hutchinson_samples < 1:
        raise ValueError("num_hutchinson_samples must be at least 1.")

    # Reuse each probe across ODE time while giving each batch element independent probes.
    z = jax.random.normal(
        rng, shape=(num_hutchinson_samples,) + observation.shape
    )

    def _log_prob_step_fn(
        t: float,
        x_t: Tuple[jax.Array, float],
        args: Tuple[PRNGKey, jax.Array, jax.Array],
    ) -> Tuple[jax.Array, jax.Array]:
        """Log prob ODE step function using Hutchinson trace estimator."""

        x_t, _ = x_t
        z, cond_observation, cond_action = args

        ode_fn = lambda x_t: _ode_fn(x_t, t, cond_observation, cond_action)
        if trace_estimator == "hutchinson":
            v_t, vjp_fn = jax.vjp(ode_fn, x_t)
            z_dot_dfdx = jax.vmap(lambda probe: vjp_fn(probe)[0])(z)
            log_prob = jnp.mean(
                jnp.sum(z_dot_dfdx * z, axis=tuple(range(1, z.ndim)))
            )
        elif trace_estimator == "exact":
            v_t = ode_fn(x_t)
            log_prob = jnp.trace(jax.jacfwd(ode_fn)(x_t))
        else:
            raise ValueError(f"Unknown trace estimator: {trace_estimator}")
        return v_t, log_prob

    def _gaussian_log_prob(y: jax.Array) -> jax.Array:
        """Standard Gaussian log prob. Assumes 1D array."""
        return -0.5 * (y.size * math.log(2 * math.pi) + jnp.sum(y**2))

    # now solve the ODE.
    def _solve(
        unbatched_x_1: jax.Array,
        unbatched_z: jax.Array,
        unbatched_cond_observation: jax.Array,
        unbatched_cond_action: jax.Array,
    ) -> jax.Array:

        term = ODETerm(_log_prob_step_fn)
        solver = Midpoint()

        solution = diffeqsolve(
            term,
            solver,
            t0=1.0,
            t1=0.0,
            dt0=-step_size,
            y0=(unbatched_x_1, 0.0),
            args=(unbatched_z, unbatched_cond_observation, unbatched_cond_action),
        )
        (x_0,), (delta_log_prob,) = solution.ys

        return _gaussian_log_prob(x_0) + delta_log_prob

    if vmap:
        log_likelihoods = jax.vmap(_solve, in_axes=(0, 1, 0, 0))(
            observation, z, cond_observation, cond_action
        )
    else:
        log_likelihoods = _solve(observation, z, cond_observation, cond_action)

    return log_likelihoods
