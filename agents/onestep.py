from functools import partial
from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from flax.training.train_state import TrainState

from utils.flax_utils import nonpytree_field
from utils.flow_modules import UNet
from utils.flow_utils import flow_sample, validate_flow_config
from utils.networks import UnconditionalEmbedding, count_parameters
from utils.types import InfoDict, PRNGKey




def _tree_l2(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(leaf**2) for leaf in leaves])))


class OneStepFlowModel(flax.struct.PyTreeNode):
    """One-step transition flow with geometric-horizon rollout sampling."""

    rng: Any
    flow: TrainState
    uncond_embed: TrainState
    config: Any = nonpytree_field()

    def _sample_available_history(self, history, rng):
        """Sample each example's suffix length and repeat its earliest state on the left."""
        batch_size, history_size = history.shape[:2]
        available = jax.random.randint(
            rng, (batch_size,), minval=1, maxval=history_size + 1
        )
        first_available = history_size - available
        positions = jnp.arange(history_size)[None, :]
        gather_idxs = jnp.maximum(positions, first_available[:, None])
        gather_idxs = gather_idxs.reshape(
            (batch_size, history_size) + (1,) * (history.ndim - 2)
        )
        gather_idxs = jnp.broadcast_to(gather_idxs, history.shape)
        return jnp.take_along_axis(history, gather_idxs, axis=1), available

    @staticmethod
    def _flatten_history(history):
        return history.reshape((history.shape[0], -1))

    def _initial_history(self, observation):
        return jnp.repeat(
            observation[:, None], int(self.config.get("history_size", 1)), axis=1
        )

    @jax.jit
    def total_loss(self, params, uncond_embed_params, batch, rng=None):
        rng = self.rng if rng is None else rng
        batch_size = batch["observations"].shape[0]
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        actions = batch["actions"]

        t_key, source_key, dropout_key, history_key = jax.random.split(rng, 4)
        t = jax.random.uniform(t_key, (batch_size,))
        source = jax.random.normal(source_key, next_observations.shape)
        t_expanded = t.reshape(
            (batch_size,) + (1,) * (next_observations.ndim - 1)
        )
        x_t = (1.0 - t_expanded) * source + t_expanded * next_observations

        history = batch.get("history_observations", observations[:, None])
        history, available_history = self._sample_available_history(
            history, history_key
        )
        history_condition = self._flatten_history(history)

        null_action = self.uncond_embed.apply_fn({"params": uncond_embed_params})
        drop_action = jax.random.bernoulli(
            dropout_key,
            p=self.config["action_dropout_prob"],
            shape=(batch_size,),
        )
        conditioned_actions = jnp.where(drop_action[:, None], null_action, actions)
        prediction = self.flow.apply_fn(
            {"params": params}, x_t, t, history_condition, conditioned_actions
        )

        target = next_observations - source

        reduction_axes = tuple(range(1, observations.ndim))
        squared_error = (prediction - target) ** 2
        if self.config["model_reduction_type"] == "SUM":
            loss = squared_error.sum(axis=reduction_axes)
        else:
            loss = squared_error.mean(axis=reduction_axes)

        dropped = drop_action.astype(loss.dtype)
        conditional = (~drop_action).astype(loss.dtype)
        info = {
            "loss": loss.mean(),
            "conditional_loss": jnp.sum(loss * conditional)
            / (jnp.sum(conditional) + 1e-8),
            "unconditional_loss": jnp.sum(loss * dropped)
            / (jnp.sum(dropped) + 1e-8),
            "action_dropout_frac": dropped.mean(),
            "history_available_mean": available_history.mean(),
            "history_full_frac": (available_history == history.shape[1]).mean(),
        }
        return loss.mean(), info

    @jax.jit
    def update(self, batch):
        rng, update_key = jax.random.split(self.rng)

        def loss_fn(params, uncond_embed_params):
            return self.total_loss(params, uncond_embed_params, batch, update_key)

        (flow_grads, uncond_grads), info = jax.grad(
            loss_fn, has_aux=True, argnums=(0, 1)
        )(self.flow.params, self.uncond_embed.params)
        info.update(
            {
                "grad/flow_l2": _tree_l2(flow_grads),
                "grad/uncond_l2": _tree_l2(uncond_grads),
            }
        )
        return (
            self.replace(
                rng=rng,
                flow=self.flow.apply_gradients(grads=flow_grads),
                uncond_embed=self.uncond_embed.apply_gradients(grads=uncond_grads),
            ),
            info,
        )

    @jax.jit
    def compute_loss(self, batch, rng=None):
        rng = self.rng if rng is None else rng
        new_rng, loss_rng = jax.random.split(rng)
        loss, info = self.total_loss(
            self.flow.params, self.uncond_embed.params, batch, rng=loss_rng
        )
        return self.replace(rng=new_rng), loss, info

    def _null_actions(self, batch_size: int) -> jax.Array:
        null_action = self.uncond_embed.apply_fn(
            {"params": self.uncond_embed.params}
        )
        return jnp.broadcast_to(null_action, (batch_size, null_action.shape[-1]))

    @partial(jax.jit, static_argnames=("use_action",))
    def sample_one_step_jnp(
        self,
        observation: jax.Array,
        action: jax.Array,
        use_action: bool = True,
    ) -> Tuple[Any, jax.Array]:
        was_unbatched = observation.ndim == 1
        if was_unbatched:
            observation = observation[None]
            action = action[None]
        history = self._initial_history(observation)
        conditioned_action = (
            action if use_action else self._null_actions(observation.shape[0])
        )
        next_observation, new_rng = flow_sample(
            self.rng,
            self.flow.apply_fn,
            self.flow.params,
            self._flatten_history(history),
            conditioned_action,
            self.config["eval_sample_step_size"],
            final_t=1.0,
            sampler_solver=self.config["eval_sampler_solver"],
            sample_template=observation,
        )
        if was_unbatched:
            next_observation = next_observation[0]
        return self.replace(rng=new_rng), next_observation

    def sample_one_step(self, observation, action, use_action: bool = True):
        model, sample = self.sample_one_step_jnp(
            jnp.asarray(observation), jnp.asarray(action), use_action=use_action
        )
        return model, np.asarray(sample)

    @partial(jax.jit, static_argnames=("max_steps",))
    def sample_rollout_jnp(self, observation, action, max_steps: int):
        """Return every future state [B, T, D], without geometric stopping."""
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if observation.ndim != 2 or action.ndim != 2:
            raise ValueError("Full rollouts require batched vector states and actions")
        # Match the geometric sampler's rollout RNG split, for pathwise tests.
        _, rollout_rng, next_rng = jax.random.split(self.rng, 3)
        history = self._initial_history(observation)
        null_actions = self._null_actions(observation.shape[0])

        def step(carry, t):
            history, rng = carry
            candidate, rng = flow_sample(
                rng, self.flow.apply_fn, self.flow.params,
                self._flatten_history(history),
                jnp.where(t == 0, action, null_actions),
                self.config["eval_sample_step_size"], final_t=1.0,
                sampler_solver=self.config["eval_sampler_solver"],
                sample_template=observation,
            )
            history = jnp.concatenate([history[:, 1:], candidate[:, None]], axis=1)
            return (history, rng), candidate

        _, states = jax.lax.scan(step, (history, rollout_rng), jnp.arange(max_steps))
        return self.replace(rng=next_rng), jnp.swapaxes(states, 0, 1)

    @partial(jax.jit, static_argnames=("max_steps",))
    def _sample_geometric_jnp(
        self,
        observation: jax.Array,
        action: jax.Array,
        discount: jax.Array,
        max_steps: int,
    ) -> Tuple[Any, jax.Array]:
        was_unbatched = observation.ndim == 1
        if was_unbatched:
            observation = observation[None]
            action = action[None]

        horizon_rng, rollout_rng, next_rng = jax.random.split(self.rng, 3)
        batch_size = observation.shape[0]
        horizons = jax.random.geometric(
            horizon_rng, p=1.0 - discount, shape=(batch_size,)
        ).astype(jnp.int32)
        horizons = jnp.minimum(horizons, max_steps)
        null_actions = self._null_actions(batch_size)
        history = self._initial_history(observation)

        def rollout_step(carry, step):
            state, history, step_rng = carry
            step_action = jnp.where(step == 0, action, null_actions)
            candidate, step_rng = flow_sample(
                step_rng,
                self.flow.apply_fn,
                self.flow.params,
                self._flatten_history(history),
                step_action,
                self.config["eval_sample_step_size"],
                final_t=1.0,
                sampler_solver=self.config["eval_sampler_solver"],
                sample_template=state,
            )
            active = step < horizons
            active = active.reshape(
                (batch_size,) + (1,) * (state.ndim - 1)
            )
            state = jnp.where(active, candidate, state)
            candidate_history = jnp.concatenate(
                [history[:, 1:], candidate[:, None]], axis=1
            )
            history_active = active[:, None]
            history = jnp.where(history_active, candidate_history, history)
            return (state, history, step_rng), None

        (sample, _, _), _ = jax.lax.scan(
            rollout_step,
            (observation, history, rollout_rng),
            jnp.arange(max_steps),
        )
        if was_unbatched:
            sample = sample[0]
        return self.replace(rng=next_rng), sample

    def sample_geometric_jnp(
        self,
        observation: jax.Array,
        action: jax.Array,
        discount: float,
        max_steps: int = None,
    ) -> Tuple[Any, jax.Array]:
        max_steps = (
            int(self.config["sample_max_steps"])
            if max_steps is None
            else int(max_steps)
        )
        if max_steps < 1:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")
        return self._sample_geometric_jnp(
            jnp.asarray(observation),
            jnp.asarray(action),
            jnp.asarray(discount),
            max_steps=max_steps,
        )

    def sample_geometric(
        self,
        observation,
        action,
        discount: float,
        max_steps: int = None,
    ):
        model, sample = self.sample_geometric_jnp(
            observation, action, discount=discount, max_steps=max_steps
        )
        return model, np.asarray(sample)

    def sample_jnp(self, observation, action):
        return self.sample_geometric_jnp(
            observation,
            action,
            discount=self.config["discount"],
            max_steps=self.config["sample_max_steps"],
        )

    def sample(self, observation, action):
        model, sample = self.sample_jnp(
            jnp.asarray(observation), jnp.asarray(action)
        )
        return model, np.asarray(sample)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        pretrained_encoder=None,
    ):
        validate_flow_config(config)
        if pretrained_encoder is not None and config.get("encode_observations", False):
            ex_observations = pretrained_encoder(ex_observations)
        if config["discrete"]:
            raise NotImplementedError("OneStepFlowModel currently supports continuous actions only.")
        if not 0.0 <= config["action_dropout_prob"] <= 1.0:
            raise ValueError("action_dropout_prob must be in [0, 1].")
        if not 0.0 <= config["discount"] < 1.0:
            raise ValueError("discount must be in [0, 1).")
        if int(config["sample_max_steps"]) < 1:
            raise ValueError("sample_max_steps must be positive.")
        if int(config.get("history_size", 1)) < 1:
            raise ValueError("history_size must be at least 1.")
        if config["encoder"] is not None:
            raise NotImplementedError("OneStepFlowModel with an internal encoder is not implemented.")

        rng = jax.random.PRNGKey(seed)
        rng, flow_key, uncond_key = jax.random.split(rng, 3)
        dummy_time = np.zeros((1,), dtype=jnp.float32)
        action_dim = ex_actions.shape[-1]
        observation_dim = ex_observations.shape[-1]
        history_condition = np.repeat(
            ex_observations[:, None], int(config.get("history_size", 1)), axis=1
        ).reshape((ex_observations.shape[0], -1))

        if config["architecture"] != "unet":
            raise ValueError("The one-step model supports architecture=unet only.")
        flow_def = UNet(
            num_stages=config["num_stages"],
            blocks_per_stage=config["blocks_per_stage"],
            cond_hidden_dims=config["cond_hidden_dims"],
            residual_hidden_dims=config["residual_hidden_dims"],
            temb_scale_factor=1000.0,
            use_batch_norm=config["use_batch_norm"],
            output_dim=observation_dim,
        )

        flow_params = flow_def.init(
            flow_key, ex_observations, dummy_time, history_condition, ex_actions
        )["params"]
        flow = TrainState.create(
            apply_fn=flow_def.apply,
            params=flow_params,
            tx=optax.adamw(
                learning_rate=config["lr"], eps=1e-4, weight_decay=config["wd"]
            ),
        )
        print(f"One-step flow parameter count: {count_parameters(flow_params)}")

        uncond_embed_def = UnconditionalEmbedding(action_dim=action_dim)
        uncond_embed_params = uncond_embed_def.init(uncond_key)["params"]
        uncond_embed = TrainState.create(
            apply_fn=uncond_embed_def.apply,
            params=uncond_embed_params,
            tx=optax.adamw(
                learning_rate=config["lr"], eps=1e-4, weight_decay=config["wd"]
            ),
        )
        return cls(
            rng=rng,
            flow=flow,
            uncond_embed=uncond_embed,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="onestep",
            loss_type="ONESTEP_CFM",
            architecture="unet",
            lr=1e-4,
            wd=1e-3,
            batch_size=1024,
            normalize_observations=True,
            norm_method="standard",
            cond_hidden_dims=(512, 512, 512),
            residual_hidden_dims=(512, 512, 512),
            num_stages=1,
            blocks_per_stage=1,
            use_batch_norm=False,
            model_reduction_type="SUM",
            train_sample_step_size=0.1,
            eval_sample_step_size=0.1,
            eval_sampler_solver="midpoint",
            action_dropout_prob=0.5,
            discount=0.99,
            sample_max_steps=200,
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            precomputed_features=False,
            encode_observations=False,
            dataset_class="OneStepHistoryGCDataset",
            history_size=1,
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
