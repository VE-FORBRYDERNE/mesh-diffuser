import haiku as hk
import jax
from jax.experimental.maps import thread_resources
import jax.numpy as jnp
import numpy as np
from mesh_transformer.util import head_print, to_bf16
from typing import Union

from .diffusion_layers import (
    Conv2D,
    apply_conv_channel_parallel,
    UNetTimeEmbeddingShard,
    UNetDownShard,
    UNetMiddleShard,
    UNetUpShard,
    UNetOutShard,
)


class UNetShard(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config

        self.center_input_sample = config.get("center_input_sample", False)
        self.block_out_channels = config["block_out_channels"]
        self.time_embedding_channels = 4 * self.block_out_channels[0]
        self.embed_dim = config["block_out_channels"][0]
        self.shards = config["cores_per_replica"]
        assert self.embed_dim % self.shards == 0
        self.embed_dim_per_shard = self.embed_dim // self.shards

        self.embed = UNetTimeEmbeddingShard(config)
        self.conv_in = Conv2D(
            self.embed_dim_per_shard, kernel_shape=3, padding=(1, 1), name="conv_in"
        )
        block_in_channels = [self.block_out_channels[0]] + self.block_out_channels[:-1]
        self.down = [
            UNetDownShard(
                config,
                layers=config["inner_layers"],
                channels_in=channels_in,
                channels_out=channels_out,
                time_embedding_channels=self.time_embedding_channels,
                heads=config["attention_head_dim"],
                activation=config.get("activation", "silu"),
                has_attention=i != len(self.block_out_channels) - 1,
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
            time_embedding_channels=self.time_embedding_channels,
            heads=config["attention_head_dim"],
            activation=config.get("activation", "silu"),
            has_attention=True,
            name="middle",
        )
        block_in_channels = self.block_out_channels[1:] + [self.block_out_channels[-1]]
        block_last_layer_residual_in_channels = [
            self.block_out_channels[0]
        ] + self.block_out_channels[:-1]
        self.up = [
            UNetUpShard(
                config,
                layers=1 + config["inner_layers"],
                last_layer_residual_channels_in=last_layer_residual_channels_in,
                channels_in=channels_in,
                channels_out=channels_out,
                time_embedding_channels=self.time_embedding_channels,
                heads=config["attention_head_dim"],
                activation=config.get("activation", "silu"),
                has_attention=i != 0,
                has_convolution=i != len(self.block_out_channels) - 1,
                name=f"up_{i}",
            )
            for i, (
                last_layer_residual_channels_in,
                channels_in,
                channels_out,
            ) in enumerate(
                zip(
                    reversed(block_last_layer_residual_in_channels),
                    reversed(block_in_channels),
                    reversed(self.block_out_channels),
                )
            )
        ]
        self.out = UNetOutShard(
            config,
            out_channels=config["out_channels"],
            activation=config.get("activation", "silu"),
            name="out",
        )

    def __call__(
        self,
        x: jnp.DeviceArray,
        y: jnp.DeviceArray,
        t: Union[float, int, jnp.DeviceArray],
    ):
        if self.center_input_sample:
            x = 2 * x - 1

        time_embeddings = self.embed(t, x.shape[0])
        x = apply_conv_channel_parallel(self.conv_in, x)

        residuals = [x]
        for module in self.down:
            down_residuals = module(x, y=y, time_embeddings=time_embeddings)
            x = down_residuals[-1]
            residuals += down_residuals

        x = self.middle(x, y=y, time_embeddings=time_embeddings)

        for module in self.up:
            down_residuals = residuals[-len(module.layers) :]
            residuals = residuals[: -len(module.layers)]
            x = module(
                x, y=y, down_residuals=down_residuals, time_embeddings=time_embeddings
            )

        return self.out(x)


class UNet2D:
    def __init__(self, config, dematerialized=False):
        self.config = config

        def init(key, x, y, t):
            def pass_to_model(x, y, t):
                model = UNetShard(config)
                return model(x, y, t)

            param_init_fn = hk.transform(
                hk.experimental.optimize_rng_use(pass_to_model)
            ).init
            params = param_init_fn(key, x, y, t)
            return {"params": to_bf16(params), "step": np.array(0), "opt_state": {}}

        self.init_xmap = jax.experimental.maps.xmap(
            fun=init,
            in_axes=(
                ["diffusion_shard", ...],
                ["diffusion_batch", ...],
                ["diffusion_batch", ...],
                ["diffusion_batch", ...],
            ),
            out_axes=["diffusion_shard", ...],
            axis_resources={"diffusion_shard": "mp", "diffusion_batch": "dp"},
        )

        key = hk.PRNGSequence(42)

        assert thread_resources.env.shape["mp"] == config["cores_per_replica"]

        dp = thread_resources.env.shape["dp"]
        mp = thread_resources.env.shape["mp"]

        mp_per_host = min(mp, 8)

        x = jax.random.uniform(
            next(key), (1, 2, config["in_channels"], 64, 64), minval=-2, maxval=2
        ).astype(jnp.float32)
        y = jax.random.uniform(next(key), (1, 2, 77, 768), minval=-2, maxval=2).astype(
            jnp.float32
        )

        head_print("key shape", jnp.array(key.take(mp_per_host)).shape)
        head_print("in shape", x.shape)
        head_print("context shape", y.shape)

        head_print("dp", dp)
        head_print("mp", mp)

        self.gen_length = 1
        self.state = self.init_xmap(
            jnp.array(key.take(mp_per_host)), x, y, jnp.array([4], dtype=jnp.float32)
        )

        param_count = hk.data_structures.tree_size(self.state["params"])
        head_print(f"Total parameters: {param_count}")
