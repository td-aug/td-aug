import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from agents.tdflow import (
    FlowModel, LossType, ModelReductionType, mc_loss_fn,
    conditional_td2_loss_fn, conditional_mc_hc_td2_loss_fn,
)
from utils.flow_modules import UNet
from utils.flow_utils import flow_sample, validate_flow_config
from utils.mchc_discounts import resolve_mchc_discounts
from utils.networks import count_parameters


class ChunkTDFlowModel(FlowModel):
    """Predict a future state conditioned on the current state and an action chunk."""

    @staticmethod
    def _flatten(sequence):
        return sequence.reshape((sequence.shape[0], -1))

    def _prepare_actions(self, actions):
        if actions.ndim > 2:
            actions = self._flatten(actions)
        return actions

    def _mc_loss(self, params, batch, rng):
        """Adapt action-chunk batches to the shared single-state MC objective."""
        return mc_loss_fn(
            params=params,
            config=self.config,
            apply_fn=self.flow.apply_fn,
            target_params=self.target_flow.params,
            target_apply_fn=self.target_flow.apply_fn,
            batch={
                "observations": batch["observations"],
                "actions": self._prepare_actions(
                    batch.get("condition_actions", batch["actions"])
                ),
                "value_goals": batch["future_observations"],
            },
            rng=rng,
            model_reduction_type=self.model_reduction_type,
        )

    def _bootstrap_loss(self, loss_fn, config, params, batch, rng):
        return loss_fn(
            params=params,
            config=config,
            apply_fn=self.flow.apply_fn,
            target_params=self.target_flow.params,
            target_apply_fn=self.target_flow.apply_fn,
            model_reduction_type=self.model_reduction_type,
            batch=batch,
            rng=rng,
        )

    def _td2_loss(self, params, batch, rng):
        losses, info = self._bootstrap_loss(
            conditional_td2_loss_fn, self.config, params,
            {
                "observations": batch["observations"],
                "actions": self._prepare_actions(batch.get("condition_actions", batch["actions"])),
                "next_observations": batch["bootstrap_observations"],
                "next_actions": jax.lax.stop_gradient(self._prepare_actions(
                    batch.get("bootstrap_actions", batch["next_actions"])
                )),
            }, rng,
        )
        info["td2_effective_discount"] = jnp.asarray(self.config["discount"])
        info["bootstrap_action_valid_frac"] = batch.get(
            "bootstrap_action_valids", jnp.ones(())).mean()
        info["bootstrap_action_prior_frac"] = batch.get(
            "bootstrap_uses_action_prior", jnp.zeros(())).mean()
        return losses, info

    def _mc_hc_td2_loss(self, params, batch, rng):
        # convert the loss discounts here.
        local_discount, final_discount = resolve_mchc_discounts(self.config)
        config = dict(self.config, discount=local_discount, final_discount=final_discount)
        losses, info = self._bootstrap_loss(
            conditional_mc_hc_td2_loss_fn, config, params,
            {
                "observations": batch["observations"],
                "actions": self._prepare_actions(batch.get("condition_actions", batch["actions"])),
                "next_observations": batch["bootstrap_observations"],
                "mc_offset_value_goals": batch["future_observations"],
                "mc_offset_value_goal_actions": jax.lax.stop_gradient(
                    self._prepare_actions(batch["future_chunk_actions"])
                ),
            }, rng,
        )
        info["mchc_effective_local_discount"] = jnp.asarray(local_discount)
        info["mchc_effective_final_discount"] = jnp.asarray(final_discount)
        info["future_bootstrap_action_valid_frac"] = batch.get(
            "future_chunk_action_valids", jnp.ones(())).mean()
        info["future_bootstrap_action_prior_frac"] = batch.get(
            "future_bootstrap_uses_action_prior", jnp.zeros(())).mean()
        return losses, info

    @jax.jit
    def total_loss(self, params, batch, rng=None):
        rng = self.rng if rng is None else rng
        if self.loss_type == LossType.MC:
            loss, info = self._mc_loss(params, batch, rng)
        elif self.loss_type == LossType.TD2_CFM:
            loss, info = self._td2_loss(params, batch, rng)
        elif self.loss_type == LossType.MC_HC_TD2:
            loss, info = self._mc_hc_td2_loss(
                params, batch, rng
            )
        else:
            raise ValueError(
                f"ChunkTDFlowModel only supports MC, TD2_CFM, and MC_HC_TD2, got {self.loss_type}."
            )
        return loss.mean(), info

    @jax.jit
    def sample_jnp(self, observation, action):
        was_unbatched = observation.ndim == 1
        if was_unbatched:
            observation, action = observation[None], action[None]
        sample, rng = flow_sample(
            self.rng,
            self.flow.apply_fn,
            self.flow.params,
            observation,
            self._prepare_actions(action),
            self.config["eval_sample_step_size"],
            final_t=1.0,
            sampler_solver=self.config["eval_sampler_solver"],
        )
        if was_unbatched:
            sample = sample[0]
        return self.replace(rng=rng), sample

    def sample(self, observation, action):
        model, sample = self.sample_jnp(jnp.asarray(observation), jnp.asarray(action))
        return model, np.asarray(sample)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, pretrained_encoder=None):
        validate_flow_config(config)
        if config.get("encode_observations", False):
            if pretrained_encoder is None:
                raise ValueError(
                    "encode_observations=True requires a pretrained encoder."
                )
            ex_observations = pretrained_encoder(ex_observations)
        if config["discrete"]:
            raise NotImplementedError("ChunkTDFlowModel supports continuous actions only.")
        if config["architecture"] != "unet":
            raise ValueError("ChunkTDFlowModel currently supports architecture='unet' only.")
        for key in ("history_size", "prediction_chunk_size"):
            if config.get(key, 1) != 1:
                raise ValueError("Chunk GHM supports one current state and one future state only.")
        action_chunk_size = int(config["action_chunk_size"])
        if action_chunk_size < 1:
            raise ValueError("action_chunk_size must be at least 1.")
        if config.get("state_representation", "full") != "full":
            raise ValueError("Chunk models require state_representation=full.")
        if config["loss_type"] not in ("MC", "TD2_CFM", "MC_HC_TD2"):
            raise ValueError(
                "ChunkTDFlowModel supports loss_type MC, TD2_CFM, or MC_HC_TD2."
            )
        if config["loss_type"] == "MC_HC_TD2":
            resolve_mchc_discounts(config)
        rng = jax.random.PRNGKey(seed)
        # Preserve flow initialization and training RNG from older checkpoints.
        rng, flow_key, _ = jax.random.split(rng, 3)
        batch_size = ex_observations.shape[0]
        observation_dim = ex_observations.shape[-1]
        if ex_actions.ndim > 2:
            ex_actions = ex_actions.reshape(batch_size, -1)
        elif action_chunk_size > 1:
            raise ValueError(
                "action_chunk_size > 1 requires example actions with shape "
                "[batch, action_chunk_size, action_dim]."
            )
        dummy_time = np.zeros((batch_size,), dtype=np.float32)
        dummy_target = np.zeros(
            (batch_size, observation_dim), dtype=np.float32
        )
        observations = ex_observations
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
            flow_key, dummy_target, dummy_time, observations, ex_actions
        )["params"]
        flow = TrainState.create(
            apply_fn=flow_def.apply,
            params=flow_params,
            tx=optax.adamw(
                learning_rate=config["lr"], eps=1e-4, weight_decay=config["wd"]
            ),
        )
        target_flow = TrainState.create(
            apply_fn=flow_def.apply,
            params=flow_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        print(f"Chunk TDFlow parameter count: {count_parameters(flow_params)}")
        return cls(
            rng=rng,
            flow=flow,
            target_flow=target_flow,
            config=flax.core.FrozenDict(**config),
            loss_type=LossType[config["loss_type"]],
            model_reduction_type=ModelReductionType[config["model_reduction_type"]],
        )


def get_config():
    from agents.tdflow import get_config as get_tdflow_config

    config = get_tdflow_config()
    config.agent_name = "chunk_tdflow"
    config.dataset_class = "ChunkTDFlowGCDataset"
    config.action_chunk_size = 1
    config.bootstrap_stride = 1
    config.mchc_sampling_unit = "primitive"
    config.supports_log_likelihood = False
    return config
