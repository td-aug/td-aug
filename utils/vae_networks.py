import flax.linen as nn
import jax
import jax.numpy as jnp


class VAEPixelEncoder(nn.Module):
    feature_dim: int
    num_layers: int = 2
    num_filters: int = 32

    def setup(self):
        self.convs = [
            nn.Conv(
                self.num_filters, kernel_size=(3, 3), strides=(2, 2), padding="VALID"
            )
        ] + (self.num_layers - 1) * [
            nn.Conv(
                self.num_filters, kernel_size=(3, 3), strides=(1, 1), padding="VALID"
            )
        ]

        self.mean_fc = nn.Dense(self.feature_dim)
        self.mean_ln = nn.LayerNorm()
        self.log_std_fc = nn.Dense(self.feature_dim)
        self.log_std_ln = nn.LayerNorm()

    @nn.compact
    def __call__(self, obs, stopgrad=False, scale_in=True):
        if scale_in:
            obs = obs / 255.0

        conv = obs
        for conv_layer in self.convs:
            conv = conv_layer(conv)
            conv = nn.relu(conv)

        h = jnp.reshape(conv, (conv.shape[0], -1))
        if stopgrad:
            h = jax.lax.stop_gradient(h)

        mean_h_fc = self.mean_fc(h)
        mean_h_norm = self.mean_ln(mean_h_fc)
        mean_out = nn.tanh(mean_h_norm)

        log_std_h_fc = self.log_std_fc(h)
        log_std_h_norm = self.log_std_ln(log_std_h_fc)
        log_std_out = nn.tanh(log_std_h_norm)

        return mean_out, log_std_out


class PixelDecoder(nn.Module):
    out_channels: int = 3
    num_layers: int = 2
    num_filters: int = 32

    def setup(self):
        self.fc = nn.Dense(self.num_filters * 29 * 29)

        self.deconvs = (self.num_layers - 1) * [
            nn.ConvTranspose(
                self.num_filters, kernel_size=(3, 3), strides=(1, 1), padding="VALID"
            )
        ] + [
            nn.ConvTranspose(
                self.out_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=[(2, 3), (2, 3)],  # can also be [(3, 2), (3, 2)] -- why?
            )
        ]

    @nn.compact
    def __call__(self, h, scale_out=False):
        h = nn.relu(self.fc(h))

        deconv = jnp.reshape(h, (-1, 29, 29, self.num_filters))
        for i, deconv_layer in enumerate(self.deconvs):
            deconv = deconv_layer(deconv)
            if i < self.num_layers - 1:
                deconv = nn.relu(deconv)

        if scale_out:
            deconv = deconv * 255.0

        return deconv
