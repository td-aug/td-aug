"""Goal-conditioned flow behavioral cloning."""

from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActorVectorField






class GCFBCAgent(flax.struct.PyTreeNode):
    """Goal-conditioned flow behavioral cloning (GCFBC) agent."""

    # from horizon reduction: https://github.com/seohongpark/horizon-reduction/blob/master/agents/gcfbc.py

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    pretrained_encoder: Any = nonpytree_field(default=None)


    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the flow BC loss."""
        batch_size = batch["actions"].shape[0]
        x_rng, t_rng, dropout_rng = jax.random.split(rng, 3)
        actor_goals = batch["actor_goals"]
        x_1 = batch["actions"]
        if self.config["action_chunk_len"] > 1:
            x_1 = x_1.reshape((batch_size, -1))
        x_0 = jax.random.normal(x_rng, x_1.shape)
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        y = x_1 - x_0
        if self.pretrained_encoder is not None:
            obs = batch["observations"]
            goals = actor_goals
            inputs = jnp.concatenate([obs, goals], axis=1)
            if self.config["encoder"] is not None:
                inputs = self.network.select("encoder")(
                    inputs,
                    train=True,
                    params=grad_params,
                    rngs={"dropout": dropout_rng},
                )
        elif self.config["encoder"] is not None:
            inputs = self.network.select("encoder")(
                batch["observations"],
                actor_goals,
                params=grad_params,
                train=True,
                rngs={"dropout": dropout_rng},
            )
        else:
            if self.config["single_input_to_actor"]:
                inputs = batch["observations"]
            else:
                inputs = jnp.concatenate(
                    [batch["observations"], actor_goals], axis=-1
                )

        pred = self.network.select("actor")(
            inputs, actions=x_t, times=t, params=grad_params
        )
        squared_error = (pred - y) ** 2
        actor_loss = jnp.mean(squared_error)

        actor_info = {"actor_loss": actor_loss}
        if "aug_mask" in batch:
            aug_mask = jnp.asarray(batch["aug_mask"], dtype=squared_error.dtype)
            if aug_mask.ndim < squared_error.ndim:
                aug_mask = jnp.expand_dims(aug_mask, axis=-1)
            aug_mask = jnp.broadcast_to(aug_mask, squared_error.shape)
            noaug_mask = 1.0 - aug_mask
            actor_info["actor_loss_aug"] = jnp.sum(squared_error * aug_mask) / (
                jnp.sum(aug_mask) + 1e-8
            )
            actor_info["actor_loss_noaug"] = jnp.sum(squared_error * noaug_mask) / (
                jnp.sum(noaug_mask) + 1e-8
            )

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        info["learning_rate"] = jnp.asarray(self.config["lr"], dtype=jnp.float32)

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("clip_actions",))
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
        clip_actions=True,
    ):
        """Sample actions from the actor."""
        # encode observations and goals if necessary
        if self.pretrained_encoder is not None:
            # add leading dimension 64,64,3 -> 1,64,64,3
            observations = jnp.expand_dims(observations, axis=0)
            goals = jnp.expand_dims(goals, axis=0)

            inputs = self.pretrained_encoder(observations, goals)

            if self.config["encoder"] is not None:
                # inputs = jnp.expand_dims(inputs, axis=0)
                inputs = self.network.select("encoder")(inputs)
        elif self.config["encoder"] is not None:
            inputs = self.network.select("encoder")(observations, goals, train=False)
        else:
            if self.config["single_input_to_actor"]:
                inputs = observations
            else:
                inputs = jnp.concatenate([observations, goals], axis=-1)

        action_dim = self.config["action_dim"]
        if self.config["action_chunk_len"] > 1:
            action_dim *= self.config["action_chunk_len"]
        actions = jax.random.normal(seed, (*inputs.shape[:-1], action_dim))

        for i in range(self.config["flow_steps"]):
            t = jnp.full((*inputs.shape[:-1], 1), i / self.config["flow_steps"])  #
            vels = self.network.select("actor")(inputs, actions=actions, times=t)
            actions = actions + vels / self.config["flow_steps"]
        if len(actions.shape) > 1:
            actions = actions.squeeze(0)
        if clip_actions:
            actions = jnp.clip(actions, -1, 1)
        if self.config["action_chunk_len"] > 1:
            actions = actions.reshape(
                self.config["action_chunk_len"], self.config["action_dim"]
            )
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        pretrained_encoder=None,
        ex_goals=None,
    ):
        if config.get("observation_history_size", 1) != 1:
            raise ValueError("GCFBC uses observation_history_size=1.")
        if config.get("unsqeeze_rep", False):
            raise ValueError("Legacy unsqeeze_rep inputs are not supported.")
        rng = jax.random.PRNGKey(seed)
        if config.get("actor_architecture", "mlp") != "mlp":
            raise ValueError("GCFBC supports only actor_architecture=mlp.")
        if config.get("cfg_enabled", False) or config.get("horizon_encoding", "none") != "none":
            raise ValueError("GCFBC does not support CFG or horizon conditioning.")
        rng, init_rng = jax.random.split(rng, 2)
        if ex_goals is None:
            ex_goals = ex_observations

        if pretrained_encoder is not None:
            if config["encode_observations"]:
                ex_observations = pretrained_encoder(ex_observations)
                ex_goals = pretrained_encoder(ex_goals)
        assert not config["discrete"]
        action_dim = ex_actions.shape[-1]
        actor_action_dim = action_dim
        if config["action_chunk_len"] > 1:
            ex_actions = ex_actions.reshape((ex_actions.shape[0], -1))
            actor_action_dim = ex_actions.shape[-1]

        ex_times = ex_actions[..., :1]  # dummy flow matching timestep

        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["actor"] = GCEncoder(
                concat_encoder=encoder_module(
                    dropout_rate=config.get("impala_encoder_dropout", 0.0),
                )
            )
            ex_input_dim = encoders["actor"].concat_encoder.mlp_hidden_dims[-1]
        else:
            if config["single_input_to_actor"]:
                ex_input_dim = ex_observations.shape[-1]
            else:
                ex_input_dim = ex_observations.shape[-1] + ex_goals.shape[-1]

        ex_input = jnp.ones(
            (1, ex_input_dim), dtype=jnp.float32
        )  # input for initialization

        actor_def = GCActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=actor_action_dim,
            layer_norm=config["layer_norm"],
        )
        network_info = dict()
        network_info["actor"] = (
            actor_def,
            (ex_input, None, ex_actions, ex_times),
        )
        if config["encoder"] is not None:
            network_info["encoder"] = (
                encoders["actor"],
                (ex_observations, ex_goals),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])

        init_rng, dropout_rng = jax.random.split(init_rng)
        network_params = network_def.init(
            {"params": init_rng, "dropout": dropout_rng}, **network_args
        )["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config["action_dim"] = action_dim
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            pretrained_encoder=pretrained_encoder,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="gcfbc",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(
                512,
                512,
                512,
            ),  # Actor network hidden dimensions.
            actor_architecture="mlp",  # Retained in saved configurations.
            normalize_observations=False,  # Whether to normalize observations.
            layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor (unused by default; can be used for sampling future goals in GCDataset).
            action_dim=ml_collections.config_dict.placeholder(
                int
            ),  # Action dimension (set automatically).
            flow_steps=10,  # Number of flow steps.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            impala_encoder_dropout=0.0,  # Dropout rate for the policy encoder.
            precomputed_features=False,  # Whether the encoder uses precomputed features, no longer used here
            encode_observations=False,  # Whether to encode observations (set automatically).
            goal_representation="full",  # Full observation or task-specific oracle goal.
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=True,  # Unused (defined for compatibility with GCDataset).
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(
                int
            ),  # Number of frames to stack.
            action_chunk_len=1,  # Number of actions to chunk when sampling from the dataset.
            single_input_to_actor=False,  # Single input (s) instead of double input (s, g).
        )
    )
    return config
