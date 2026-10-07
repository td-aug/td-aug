from enum import Enum, auto
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
from utils.flow_utils import (
    flow_log_likelihood,
    flow_sample,
    validate_flow_config,
)
from utils.networks import count_parameters
from utils.types import ApplyFn, InfoDict, Params, PRNGKey

class LossType(Enum):
    # Preserve the numeric identifiers used by existing checkpoints.
    TD2_CFM = 3
    MC_HC_TD2 = 4
    MC = 5


class ModelReductionType(Enum):
    SUM = auto()
    MEAN = auto()


def mc_loss_fn(
    params: Params,
    config,
    apply_fn: ApplyFn,
    target_params: Params,
    target_apply_fn: ApplyFn,
    batch: dict,
    rng: PRNGKey,
    model_reduction_type: ModelReductionType = ModelReductionType.SUM,
) -> Tuple[jax.Array, InfoDict]:


    batch_size = batch["observations"].shape[0]
    observations = batch["observations"]
    value_goals = batch["value_goals"]
    # Preserve the noise keys used before removing action dropout.
    t_key, source_key, *_ = jax.random.split(rng, 5)
    t = jax.random.uniform(t_key, (batch_size,))

    # Sample the initial Gaussian noise.
    x_0 = jax.random.normal(source_key, observations.shape)

    actions = batch["actions"]

    # get samples MC geometric samples
    x_1 = value_goals

    # generate mixture target for x_t, x_t = (1 - t) * x_0 + t * x_1
    t_expanded = t.reshape((t.shape[0],) + (1,) * (observations.ndim - 1))
    x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1

    pred_t = apply_fn({"params": params}, x_t, t, observations, actions)
    pred_dx_t = pred_t
    dx_t = x_1 - x_0
    squares = (pred_dx_t - dx_t) ** 2

    axis = tuple(range(1, observations.ndim))
    if model_reduction_type == ModelReductionType.SUM:
        loss = squares.sum(axis=axis)
    else:
        loss = squares.mean(axis=axis)
    return loss, {
        "loss": loss.mean(),
    }


def conditional_mc_hc_td2_loss_fn(
    params: Params,
    config: Any,
    apply_fn: ApplyFn,
    target_params: Params,
    target_apply_fn: ApplyFn,
    model_reduction_type: ModelReductionType,
    batch: dict,
    rng: PRNGKey,
) -> Tuple[jax.Array, InfoDict]:
    """Returns the (unreduced) TD^2-CFM loss function."""
    batch_size = batch["observations"].shape[0]
    observations = batch["observations"]
    actions = batch["actions"]
    next_observations = batch["next_observations"]
    future_observations = batch["mc_offset_value_goals"]
    future_actions = batch["mc_offset_value_goal_actions"]

    discount = config["final_discount"]
    assert 0.0 <= config["discount"] < 1.0, "config['discount'] must be in [0, 1)."
    assert 0.0 <= discount < 1.0, "config['final_discount'] must be in [0, 1)."
    local_discount = config["discount"]
    assert (
        discount >= local_discount
    ), "MC_HC_TD2 expects final_discount >= local_discount."

    t_key, single_step_source_key, mc_source_key, bootstrap_source_key = jax.random.split(
        rng, 4
    )
    t = jax.random.uniform(t_key, (batch_size,))

    # compute single-step target term
    single_step_x_0 = jax.random.normal(single_step_source_key, observations.shape)

    # expand to to match observation shape
    t_expanded = t.reshape((t.shape[0],) + (1,) * (observations.ndim - 1))
    single_step_x_t = (
        1.0 - t_expanded
    ) * single_step_x_0 + t_expanded * next_observations

    # now actually compute the loss
    single_step_pred_t = apply_fn(
        {"params": params},
        single_step_x_t,
        t,
        observations,
        actions,
    )
    single_step_dx_t = next_observations - single_step_x_0
    single_step_pred_dx_t = single_step_pred_t

    assert single_step_x_0.shape == next_observations.shape

    axis = tuple(range(1, observations.ndim))
    if model_reduction_type == ModelReductionType.SUM:
        single_step_loss = ((single_step_pred_dx_t - single_step_dx_t) ** 2).sum(
            axis=axis
        )
    else:
        single_step_loss = ((single_step_pred_dx_t - single_step_dx_t) ** 2).mean(
            axis=axis
        )

    # now integrate target vector field to `t` to get target samples.
    num_train_steps = 1.0 / config["train_sample_step_size"]
    sample_step_sizes = t / num_train_steps

    # second term, MC loss on empirical future MC samples (effectively MC loss)
    mc_x_0 = jax.random.normal(mc_source_key, observations.shape)

    mc_x_t = (1.0 - t_expanded) * mc_x_0 + t_expanded * future_observations

    # compute loss
    mc_pred_t = apply_fn(
        {"params": params},
        mc_x_t,
        t,
        observations,
        actions,
    )
    mc_dx_t = future_observations - mc_x_0
    mc_pred_dx_t = mc_pred_t
    assert mc_x_0.shape == future_observations.shape
    if model_reduction_type == ModelReductionType.SUM:
        mc_loss = ((mc_pred_dx_t - mc_dx_t) ** 2).sum(axis=axis)
    else:
        mc_loss = ((mc_pred_dx_t - mc_dx_t) ** 2).mean(axis=axis)

    
    # third term, HC loss
    mc_hc_bootstrap_x_t, _ = flow_sample(
        bootstrap_source_key,
        target_apply_fn,
        target_params,
        future_observations,
        future_actions,
        sample_step_sizes,
        final_t=t,
    )
    assert mc_hc_bootstrap_x_t.shape == batch["observations"].shape


    mc_hc_bootstrap_pred_t = apply_fn(
        {"params": params},
        mc_hc_bootstrap_x_t,
        t,
        observations,
        actions,
    )
    mc_hc_bootstrap_t = target_apply_fn(
        {"params": target_params},
        mc_hc_bootstrap_x_t,
        t,
        future_observations,
        future_actions,
    )

    if model_reduction_type == ModelReductionType.SUM:
        mc_hc_bootstrap_loss = ((mc_hc_bootstrap_pred_t - mc_hc_bootstrap_t) ** 2).sum(axis=axis)
    else:
        mc_hc_bootstrap_loss = ((mc_hc_bootstrap_pred_t - mc_hc_bootstrap_t) ** 2).mean(axis=axis)

    single_step_weight = 1.0 - discount
    mc_weight = discount * (1.0 - discount) / (1.0 - local_discount)
    final_weight = discount * (discount - local_discount) / (1.0 - local_discount)


    full_loss = (
        single_step_weight * single_step_loss
        + mc_weight * mc_loss
        + final_weight * mc_hc_bootstrap_loss
    )


    return full_loss, {
        "loss": full_loss.mean(),
        "single_step_loss": single_step_loss.mean(),
        "mc_loss": mc_loss.mean(),
        "mc_hc_bootstrap_loss": mc_hc_bootstrap_loss.mean(),
    }


def conditional_td2_loss_fn(
    params: Params,
    config: Any,
    apply_fn: ApplyFn,
    target_params: Params,
    target_apply_fn: ApplyFn,
    model_reduction_type: ModelReductionType,
    batch: dict,
    rng: PRNGKey,
) -> Tuple[jax.Array, InfoDict]:
    """Returns the (unreduced) TD^2-CFM loss function."""
    observations = batch["observations"]
    next_observations = batch["next_observations"]
    next_actions = batch["next_actions"]
    discount = config["discount"]

    # set up RNG keys for source noise distribution
    batch_size = observations.shape[0]

    t_key, single_step_source_key, bootstrap_source_key, _ = jax.random.split(
        rng, 4
    )
    t = jax.random.uniform(t_key, (batch_size,))

    actions = batch["actions"]

    # compute single-step target term
    single_step_x_0 = jax.random.normal(single_step_source_key, observations.shape)

    # expand to to match observation shape
    t_expanded = t.reshape((t.shape[0],) + (1,) * (observations.ndim - 1))
    single_step_x_t = (
        1.0 - t_expanded
    ) * single_step_x_0 + t_expanded * next_observations

    # now integrate target vector field to `t` to get target samples.
    num_train_steps = 1.0 / config["train_sample_step_size"]
    sample_step_sizes = t / num_train_steps
    bootstrap_x_t, _ = flow_sample(
        bootstrap_source_key,
        target_apply_fn,
        target_params,
        next_observations,
        next_actions,
        sample_step_sizes,
        final_t=t,
    )
    assert bootstrap_x_t.shape == batch["observations"].shape

    # set up variables and target variables for forward pass
    variables = {"params": params}
    target_variables = {"params": target_params}

    # now actually compute the loss
    single_step_pred_t = apply_fn(
        variables,
        single_step_x_t,
        t,
        observations,
        actions,
    )
    single_step_dx_t = next_observations - single_step_x_0
    single_step_pred_dx_t = single_step_pred_t

    assert single_step_x_0.shape == next_observations.shape

    axis = tuple(range(1, observations.ndim))
    if model_reduction_type == ModelReductionType.SUM:
        single_step_loss = ((single_step_pred_dx_t - single_step_dx_t) ** 2).sum(
            axis=axis
        )
    else:
        single_step_loss = ((single_step_pred_dx_t - single_step_dx_t) ** 2).mean(
            axis=axis
        )

    bootstrap_pred_t = apply_fn(
        variables,
        bootstrap_x_t,
        t,
        observations,
        actions,
    )
    bootstrap_t = target_apply_fn(
        target_variables,
        bootstrap_x_t,
        t,
        next_observations,
        next_actions,
    )

    if model_reduction_type == ModelReductionType.SUM:
        bootstrap_loss = ((bootstrap_pred_t - bootstrap_t) ** 2).sum(axis=axis)
    else:
        bootstrap_loss = ((bootstrap_pred_t - bootstrap_t) ** 2).mean(axis=axis)

    full_loss = (1.0 - discount) * single_step_loss + discount * bootstrap_loss

    return full_loss, {
        "loss": full_loss.mean(),
        "single_step_loss": single_step_loss.mean(),
        "bootstrap_loss": bootstrap_loss.mean(),
    }


LOSS_FN = {
    LossType.TD2_CFM: conditional_td2_loss_fn,
    LossType.MC_HC_TD2: conditional_mc_hc_td2_loss_fn,
    LossType.MC: mc_loss_fn,
}


class FlowModel(flax.struct.PyTreeNode):
    """Temporal difference flow model."""

    rng: Any
    flow: Any
    target_flow: Any
    config: Any = nonpytree_field()
    loss_type: LossType = nonpytree_field()
    model_reduction_type: ModelReductionType = nonpytree_field()

    @jax.jit
    def total_loss(self, params, batch, rng=None):
        rng = rng if rng is not None else self.rng
        loss_impl = LOSS_FN[self.loss_type]

        loss, info = loss_impl(
            params,
            apply_fn=self.flow.apply_fn,
            target_params=self.target_flow.params,
            target_apply_fn=self.target_flow.apply_fn,
            batch=batch,
            rng=rng,
            config=self.config,
            model_reduction_type=self.model_reduction_type,
        )
        return loss.mean(), info

    @jax.jit
    def update(self, batch):
        rng, update_key = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(params, batch, update_key)

        grads, info = jax.grad(loss_fn, has_aux=True)(
            self.flow.params
        )

        def _tree_max(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            return jnp.max(jnp.stack([jnp.max(leaf) for leaf in leaves]))

        def _tree_min(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            return jnp.min(jnp.stack([jnp.min(leaf) for leaf in leaves]))

        def _tree_l2(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            return jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(leaf**2) for leaf in leaves])))

        info.update(
            {
                "grad/flow_max": _tree_max(grads),
                "grad/flow_min": _tree_min(grads),
                "grad/flow_l2": _tree_l2(grads),
            }
        )
        new_flow = self.flow.apply_gradients(grads=grads)

        # update target flow
        target_params = optax.incremental_update(
            new_flow.params, self.target_flow.params, self.config["tau"]
        )
        new_target = self.target_flow.replace(params=target_params)

        return (
            self.replace(
                rng=rng,
                flow=new_flow,
                target_flow=new_target,
            ),
            info,
        )

    # used for evaluation
    @jax.jit
    def compute_loss(self, batch, rng=None):
        rng = self.rng if rng is None else rng
        new_rng, loss_rng = jax.random.split(rng)
        loss, info = self.total_loss(
            self.flow.params, batch, rng=loss_rng
        )
        return self.replace(rng=new_rng), loss, info

    def compute_log_likelihood(
        self,
        observations: np.ndarray,
        source_observations: np.ndarray,
        source_actions: np.ndarray,
        trace_estimator: str = "hutchinson",
        step_size: float = None,
        num_hutchinson_samples: int = 1,
    ):
        """Computes log likelihood of observations `observations` given `source_observations`, `source_actions`."""

        rng, sample_key = jax.random.split(self.rng)

        # for log likelihood computation, we sample from the LL ODE with a much smaller sample size.
        log_likelihood = flow_log_likelihood(
            sample_key,
            self.flow.apply_fn,
            self.flow.params,
            observations,
            source_observations,
            source_actions,
            step_size=self.config["ll_step_size"] if step_size is None else step_size,
            trace_estimator=trace_estimator,
            num_hutchinson_samples=num_hutchinson_samples,
        )
        return self.replace(rng=rng), np.asarray(log_likelihood)

    def sample(
        self, observation: np.ndarray, action: np.ndarray
    ) -> Tuple[Any, np.ndarray]:
        """Samples from the flow model by solving the ODE."""

        x_1, new_rng = flow_sample(
            self.rng,
            self.flow.apply_fn,
            self.flow.params,
            observation,
            action,
            self.config["eval_sample_step_size"],
            final_t=1.0,
            sampler_solver=self.config.get("eval_sampler_solver", "midpoint"),
        )
        return self.replace(rng=new_rng), np.asarray(x_1)

    def sample_jnp(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[Any, jnp.ndarray]:
        """Samples from the flow model by solving the ODE."""

        x_1, new_rng = flow_sample(
            self.rng,
            self.flow.apply_fn,
            self.flow.params,
            observation,
            action,
            self.config["eval_sample_step_size"],
            final_t=1.0,
            sampler_solver=self.config.get("eval_sampler_solver", "midpoint"),
        )
        return self.replace(rng=new_rng), x_1




    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        pretrained_encoder=None,
    ):
        if config.get("discrete", False):
            raise ValueError("Only continuous actions are supported.")
        validate_flow_config(config)
        if config["loss_type"] not in ("MC", "TD2_CFM", "MC_HC_TD2"):
            raise ValueError("Supported flow losses are MC, TD2_CFM, and MC_HC_TD2.")
        rng = jax.random.PRNGKey(seed)
        # Preserve flow initialization and training RNG from older checkpoints.
        rng, flow_key, _ = jax.random.split(rng, 3)
        if pretrained_encoder is not None:
            if config.get("encode_observations", False):
                print(
                    "Warning: FlowModel with pretrained encoder expected precomputed features."
                )
                ex_observations = pretrained_encoder(ex_observations)


        dummy_time = np.zeros((1,), dtype=jnp.float32)

        if config["encoder"] is not None:
            raise NotImplementedError("FlowModel with encoder not implemented yet.")

        if config["architecture"] != "unet":
            raise ValueError("FlowModel supports architecture=unet only.")
        flow_def = UNet(
            num_stages=config["num_stages"],
            blocks_per_stage=config["blocks_per_stage"],
            cond_hidden_dims=config["cond_hidden_dims"],
            residual_hidden_dims=config["residual_hidden_dims"],
            temb_scale_factor=1000.0,
            use_batch_norm=config["use_batch_norm"],
        )

        flow_params = flow_def.init(
            flow_key, ex_observations, dummy_time, ex_observations, ex_actions
        )["params"]

        flow = TrainState.create(
            apply_fn=flow_def.apply,
            params=flow_params,
            tx=optax.adamw(
                learning_rate=config["lr"], eps=1e-4, weight_decay=config["wd"]
            ),
        )
        parameter_count = count_parameters(flow_params)
        print(f"Flow model parameter count: {parameter_count}")
        target_flow = TrainState.create(
            apply_fn=flow_def.apply,
            params=flow_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # create enum from str for loss type
        loss_type = LossType[config["loss_type"]]
        model_reduction_type = ModelReductionType[config["model_reduction_type"]]

        return cls(
            rng,
            flow=flow,
            target_flow=target_flow,
            loss_type=loss_type,
            model_reduction_type=model_reduction_type,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="tdflow",  # Agent name.
            architecture="unet",
            lr=3e-4,  # Learning rate.
            wd=1e-3,  # Weight decay.
            batch_size=1024,  # Batch size.
            normalize_observations=False,  # Whether to normalize observations.
            norm_method="standard",  # Normalization method: "standard" or "img".
            # UNet Hyperparameters.
            cond_hidden_dims=(512, 512, 512),
            residual_hidden_dims=(512, 512, 512),  # Residual network hidden dimensions.
            num_stages=1,  # Number of UNet stages.
            blocks_per_stage=1,  # Number of residual blocks per UNet stage.
            use_batch_norm=False,  # Whether to use batch normalization in the UNet.
            loss_type="TD2_CFM",
            model_reduction_type="SUM",
            train_sample_step_size=0.1,  # Step size for flow sampling during training.
            eval_sample_step_size=0.1,  # Step size for flow sampling during evaluation.
            eval_sampler_solver="midpoint",  # ODE solver for evaluation sampling ("midpoint" or "euler").
            tau=0.001,  # Target network update rate.
            discount=0.99,  # MC sampling discount or TD target weight.
            final_discount=0.0,  # Final successor discount for MC-HC.
            bootstrap_continuation_mode="recorded",  # Or policy_prior (MCHC only).
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            precomputed_features=False,  # Whether the encoder uses precomputed features.
            encode_observations=False,  # Encode observations with a pretrained encoder.
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=True,  # Used for MC flows to sample goals geometrically.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(
                int
            ),  # Number of frames to stack.
            ll_step_size=0.02,
        )
    )
    return config
