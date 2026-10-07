from typing import Any

import distrax
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.vae_networks import PixelDecoder, VAEPixelEncoder


class VAE(flax.struct.PyTreeNode):
    """Variational autoencoder."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def loss(self, batch, grad_params, rng):
        """VAE loss."""
        mean_h, log_std_h = self.network.select("encoder")(
            batch["observations"], params=grad_params, scale_in=self.config["scale_in"]
        )
        if not self.config["deterministic"]:
            std_h = jnp.exp(log_std_h)
        else:
            std_h = jnp.zeros_like(log_std_h)

        h_dist = distrax.MultivariateNormalDiag(loc=mean_h, scale_diag=std_h)
        unit_gaussian = distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(mean_h), scale_diag=jnp.ones_like(log_std_h)
        )

        # rec loss
        z = h_dist.sample(seed=rng)
        if self.config["normalize_observations"]:
            rec_obs = self.network.select("decoder")(
                z, params=grad_params, scale_out=False
            )
            rec_loss = jnp.mean((batch["observations"] - rec_obs) ** 2)
        elif self.config["scale_out"]:
            rec_obs = self.network.select("decoder")(
                z, params=grad_params, scale_out=True
            )
            rec_loss = jnp.mean((batch["observations"] - rec_obs) ** 2)
        else:
            rec_obs = self.network.select("decoder")(
                z, params=grad_params, scale_out=False
            )
            rec_loss = jnp.mean((batch["observations"] / 255.0 - rec_obs) ** 2)

        # kl loss
        if not self.config["deterministic"]:
            kl_loss = jnp.mean(h_dist.kl_divergence(unit_gaussian))
        else:
            kl_loss = 0.0

        loss = rec_loss + self.config["beta"] * kl_loss
        return loss, {
            "rec_loss": rec_loss,
            "kl_loss": kl_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        vae_loss, vae_info = self.loss(batch, grad_params, actor_rng)
        for k, v in vae_info.items():
            info[f"vae/{k}"] = v

        loss = vae_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_loss(self, batch, rng=None):
        rng = self.rng if rng is None else rng
        new_rng, loss_rng = jax.random.split(rng)
        loss, info = self.total_loss(batch, self.network.params, rng=loss_rng)
        return self.replace(rng=new_rng), loss, info

    @jax.jit
    def sample(self, observations, seed=None):
        batch_size = observations.shape[0]
        z = jax.random.normal(seed, (batch_size, self.config["feature_dim"]))
        out = self.network.select("decoder")(
            z, scale_out=self.config["scale_out"]
        )  # keep manual
        return out

    # deterministic encoding for downstream tasks
    @jax.jit
    def encode(self, observations, seed=None):
        mean_h, log_std_h = self.network.select("encoder")(
            observations, scale_in=self.config["scale_in"]
        )
        z = mean_h
        return z

    @jax.jit
    def decode(self, latents):
        reconstructions = self.network.select("decoder")(
            latents, scale_out=self.config["scale_out"]
        )
        return reconstructions

    @jax.jit
    def __call__(self, observations, goals=None, seed=None):
        stacked = (
            jnp.concatenate([observations, goals], axis=0)
            if goals is not None
            else observations
        )
        mean_h, log_std_h = self.network.select("encoder")(
            stacked, scale_in=self.config["scale_in"]
        )
        z = mean_h
        if goals is not None:
            # this is (B * 2, dim)
            # want to go to (B, 2*dim)
            z_obs, z_goals = jnp.split(z, 2, axis=0)
            return jnp.concatenate([z_obs, z_goals], axis=-1)
        else:
            return z

    # determinsitic encode-decode for evaluating reconstruction quality
    @jax.jit
    def encode_decode(self, observations, seed=None):
        z = self.encode(observations, seed=seed)
        reconstructions = self.decode(z)
        return reconstructions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        for key in ("action_loss", "latent_pred_loss", "latent_recon_loss"):
            if config.get(key, 0.0) != 0.0:
                raise ValueError(f"{key} is not supported.")
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_latents = jnp.zeros((1, config["feature_dim"]), dtype=jnp.float32)
        if config["normalize_observations"]:
            assert (
                not config["scale_out"] and not config["scale_in"]
            ), "do not scale w/ observation normalization."

        encoder_def = VAEPixelEncoder(
            feature_dim=config["feature_dim"],
            num_layers=config["num_layers"],
            num_filters=config["num_filters"],
        )
        decoder_def = PixelDecoder(
            out_channels=ex_observations.shape[-1],  # this should be channel dim
            num_layers=config["num_layers"],
            num_filters=config["num_filters"],
        )

        network_info = dict(
            encoder=(encoder_def, (ex_observations,)),
            decoder=(decoder_def, (ex_latents,)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="vae",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            feature_dim=50,  # Feature dimension.
            num_layers=2,  # Number of convolutional layers.
            num_filters=32,  # Number of convolutional filters.
            beta=1.0,  # KL coefficient for VAE loss.
            discrete=False,  # Whether the action space is discrete.
            scale_out=False,  # Whether to scale decoder output.
            scale_in=True,  # Whether to scale encoder input.
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
            actor_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            actor_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            actor_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            actor_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(
                int
            ),  # Number of frames to stack.
            normalize_observations=False,  # Whether to normalize observations.
            norm_method="img",  # Normalization method: "standard" or "img".
            loss_type=None,  # Unused.
            deterministic=False,  # Whether the encoder output is deterministic (std = 0).
        )
    )
    return config
