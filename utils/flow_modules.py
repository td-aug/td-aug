import math
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.networks import MLP, default_init

zero_init = nn.initializers.zeros_init


class UNet(nn.Module):
    num_stages: int
    blocks_per_stage: int = 1
    cond_hidden_dims: Sequence[int] = (512, 512, 512)
    residual_hidden_dims: Sequence[int] = (512, 512, 512)
    activation: Callable[[jax.Array], jax.Array] = jax.nn.mish
    temb_scale_factor: float = 1.0
    use_batch_norm: bool = False
    output_dim: Optional[int] = None

    @nn.compact
    def __call__(
        self, x: jax.Array, t: jax.Array, observations: jax.Array, actions: jax.Array
    ) -> jax.Array:
        output_dim = observations.shape[-1] if self.output_dim is None else self.output_dim

        # observation-action embedding
        obs_actions = jnp.concatenate([observations, actions], axis=-1)

        obs_action_embeds = MLP(
            hidden_dims=self.cond_hidden_dims,
            activations=self.activation,
            activate_final=False,
        )(obs_actions)

        # time embedding
        timestep_embeds = TimestepEmbedding(scale_factor=self.temb_scale_factor)(t)

        # concatenate the embeddings to produce full (s, a, t) conditioning
        cond = jnp.concatenate([obs_action_embeds, timestep_embeds], axis=-1)

        # now pass through UNet (down, middle, up)

        # `down`` blocks
        xs = []
        for _ in range(self.num_stages):
            for _ in range(self.blocks_per_stage):
                x = ResidualBlock(
                    hidden_dims=self.residual_hidden_dims,
                    activation=self.activation,
                )(x, cond)
            xs.append(x)

        # `middle` blocks
        for _ in range(self.blocks_per_stage):
            x = ResidualBlock(
                hidden_dims=self.residual_hidden_dims,
                activation=self.activation,
            )(x, cond)

        # `up` blocks
        for _ in range(self.num_stages):
            x_cond = jnp.concatenate([x, xs.pop()], axis=-1)

            for _ in range(self.blocks_per_stage):
                x = ResidualBlock(
                    hidden_dims=self.residual_hidden_dims[::-1],
                    activation=self.activation,
                )(x_cond, cond)

        assert not xs
        # final dense -> LN -> act
        x = nn.Dense(self.residual_hidden_dims[-1], kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)

        # project back to output size. should be observation dim.
        x = nn.Dense(output_dim, kernel_init=zero_init())(x)

        return x


class TimestepEmbedding(nn.Module):
    embedding_size: int = 256
    max_period: int = 10000
    hidden_dims: Sequence[int] = (256, 256)
    scale_factor: float = 1.0

    def timestep_embedding(self, t: jax.Array) -> jax.Array:
        half = self.embedding_size // 2
        freqs = jnp.exp(
            -math.log(self.max_period)
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / half
        )

        # turn into 1D array
        is_input_constant = t.ndim == 0
        if is_input_constant:
            t = jnp.array((t,))

        # scale by scale factor
        t *= self.scale_factor

        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

        if is_input_constant:
            return jnp.squeeze(embedding, axis=0)

        return embedding

    @nn.compact
    def __call__(self, t: jax.Array) -> jax.Array:
        t_emb = self.timestep_embedding(t)

        # pass through MLP now.
        t_emb = MLP(
            hidden_dims=self.hidden_dims, activations=jax.nn.mish, activate_final=False
        )(t_emb)
        return t_emb


class ResidualBlock(nn.Module):
    hidden_dims: Sequence[int] = (512, 512, 512)
    activation: Callable[[jax.Array], jax.Array] = jax.nn.mish

    @nn.compact
    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        h = x

        # MLP stack
        for i, dim in enumerate(self.hidden_dims):
            h = nn.Dense(dim, kernel_init=default_init())(h)

            if i == 0:
                h = nn.LayerNorm(use_scale=False, use_bias=False)(h)
                h = FiLM(dim)(h, cond)
            else:
                h = nn.LayerNorm()(h)

            h = self.activation(h)

        # shortcut
        if x.shape[-1] != h.shape[-1]:
            x = nn.Dense(h.shape[-1], kernel_init=default_init())(x)

        return h + x


class FiLM(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        if x.ndim > 1:
            assert cond.ndim == x.ndim
            assert x.shape[-1] == self.dim and x.shape[0] == cond.shape[0]

        film_output = nn.Dense(2 * self.dim, kernel_init=zero_init())(cond)
        gamma, beta = jnp.split(film_output, 2, axis=-1)

        return (1.0 + gamma) * x + beta
