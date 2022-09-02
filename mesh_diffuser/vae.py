import haiku as hk

from .diffusion_layers import (
    Conv2D,
    apply_conv_channel_parallel,
    UNetDownShard,
    UNetMiddleShard,
    UNetUpShard,
    UNetOutShard,
)


class EncoderShard(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)

        self.shards = config["cores_per_replica"]
        self.block_out_channels = config["block_out_channels"]
        assert self.block_out_channels[0] % self.shards == 0

        self.conv_in = Conv2D(
            self.block_out_channels[0] // self.shards,
            kernel_shape=3,
            padding=(1, 1),
            name="conv_in",
        )
        block_in_channels = [self.block_out_channels[0]] + self.block_out_channels[:-1]
        self.down = [
            UNetDownShard(
                config,
                layers=config["inner_layers"],
                channels_in=channels_in,
                channels_out=channels_out,
                time_embedding_channels=None,
                has_attention=False,
                has_convolution=i != len(self.block_out_channels) - 1,
                name=f"down_{i}",
            )
            for i, (channels_in, channels_out) in enumerate(
                zip(block_in_channels, self.block_out_channels)
            )
        ]
        self.middle = UNetMiddleShard(
            config,
            channels_out=self.block_out_channels[-1],
            time_embedding_channels=None,
            simple=True,
            has_qkv_bias=True,
            name="middle",
        )
        self.out = UNetOutShard(
            config,
            out_channels=2 * config["out_channels"],
            activation=config.get("activation", "silu"),
            name="out",
        )

    def __call__(self, x):
        x = apply_conv_channel_parallel(self.conv_in, x)
        for module in self.down:
            x = module(x)[-1]
        x = self.middle(x)
        return self.out(x)


class DecoderShard(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)

        self.shards = config["cores_per_replica"]
        self.block_out_channels = config["block_out_channels"]
        assert self.block_out_channels[-1] % self.shards == 0

        self.conv_in = Conv2D(
            self.block_out_channels[-1] // self.shards,
            kernel_shape=3,
            padding=(1, 1),
            name="conv_in",
        )
        self.middle = UNetMiddleShard(
            config,
            channels_out=self.block_out_channels[-1],
            time_embedding_channels=None,
            simple=True,
            has_qkv_bias=True,
            name="middle",
        )
        block_in_channels = self.block_out_channels[1:] + [self.block_out_channels[-1]]
        self.up = [
            UNetUpShard(
                config,
                layers=1 + config["inner_layers"],
                channels_in=channels_in,
                channels_out=channels_out,
                time_embedding_channels=None,
                has_attention=False,
                has_convolution=i != len(self.block_out_channels) - 1,
                name=f"up_{i}",
            )
            for i, (channels_in, channels_out) in enumerate(
                zip(reversed(block_in_channels), reversed(self.block_out_channels))
            )
        ]
        self.out = UNetOutShard(
            config,
            out_channels=2 * config["out_channels"],
            activation=config.get("activation", "silu"),
            name="out",
        )

    def __call__(self, x):
        x = apply_conv_channel_parallel(self.conv_in, x)
        x = self.middle(x)
        for module in self.up:
            x = module(x)
        return self.out(x)


class AutoencoderShard(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config
        self.shards = config["cores_per_replica"]
        self.latent_channels = config["latent_channels"]

        self.encoder = EncoderShard(config)
        self.quant_conv = Conv2D(
            2 * self.latent_channels, kernel_shape=1, padding=(0, 0)
        )
        self.post_quant_conv = Conv2D(
            self.latent_channels, kernel_shape=1, padding=(0, 0)
        )
        self.decoder = DecoderShard(config)

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_conv(x)
        return x

    def decode(self, x):
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        return x
