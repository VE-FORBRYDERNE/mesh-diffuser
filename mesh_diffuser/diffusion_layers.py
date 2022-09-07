import haiku as hk
from haiku._src import utils as hk_utils
from haiku._src.conv import to_dimension_numbers
import jax
import jax.numpy as jnp
import numpy as np
from mesh_transformer.layers import getactfn
from typing import Optional, Sequence, Tuple, Union


class ReplicatedLayerNorm(hk.Module):
    def __init__(self, offset=True, name=None):
        super().__init__(name)
        self.offset = offset

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.var(inputs, axis=-1, keepdims=True)

        param_shape = inputs.shape[-1:]
        scale = hk.get_parameter("scale", param_shape, inputs.dtype, init=jnp.ones)
        scale = jax.lax.all_gather(scale, "diffusion_shard")[0]

        offset = hk.get_parameter("offset", param_shape, inputs.dtype, init=jnp.zeros)
        offset = jax.lax.all_gather(offset, "diffusion_shard")[0]

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        inv = scale * jax.lax.rsqrt(variance + 1e-5)
        if self.offset:
            return inv * (inputs - mean) + offset
        else:
            return inv * (inputs - mean)


class ReplicatedGroupNorm(hk.GroupNorm):
    def __init__(
        self,
        *args,
        create_scale=True,
        create_offset=True,
        data_format="channels_first",
        **kwargs,
    ):
        super().__init__(
            *args,
            create_scale=False,
            create_offset=False,
            data_format=data_format,
            **kwargs,
        )
        self._diffusion_layers_create_scale = create_scale
        self._diffusion_layers_create_offset = create_offset
        if create_scale:
            self._diffusion_layers_scale_init = (
                kwargs["scale_init"]
                if kwargs.get("scale_init", None) is not None
                else jnp.ones
            )
        if create_offset:
            self._diffusion_layers_offset_init = (
                kwargs["offset_init"]
                if kwargs.get("offset_init", None) is not None
                else jnp.zeros
            )

    def __call__(self, x, scale=None, offset=None, **kwargs):
        dtype = x.dtype
        if self.rank is None:
            channels = x.shape[self.channel_index]
            self._initialize(x, channels)
        if self.channel_index == -1:
            params_shape = (x.shape[-1],)
        else:
            assert self.channel_index == 1
            params_shape = (x.shape[1],)
        if self._diffusion_layers_create_scale:
            scale = jax.lax.all_gather(
                hk.get_parameter(
                    "scale", params_shape, dtype, self._diffusion_layers_scale_init
                ),
                "diffusion_shard",
            )[0]
        if self._diffusion_layers_create_offset:
            offset = jax.lax.all_gather(
                hk.get_parameter(
                    "offset", params_shape, dtype, self._diffusion_layers_offset_init
                ),
                "diffusion_shard",
            )[0]
        if self.channel_index != -1:
            scale = scale.reshape(scale.shape + (1,) * (self.rank - 2))
            offset = offset.reshape(offset.shape + (1,) * (self.rank - 2))
        return super().__call__(x, scale=scale, offset=offset, **kwargs)


class ConvND(hk.Module):
    """General N-dimensional convolutional. (from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/conv.py)"""

    def __init__(
        self,
        num_spatial_dims: int,
        output_channels: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[
            str,
            Tuple[int, int],
            Sequence[Tuple[int, int]],
            hk.pad.PadFn,
            Sequence[hk.pad.PadFn],
        ] = "SAME",
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        data_format: str = "channels_last",
        mask: Optional[jnp.ndarray] = None,
        feature_group_count: int = 1,
        name: Optional[str] = None,
    ):
        """Initializes the module.
        Args:
            num_spatial_dims: The number of spatial dimensions of the input.
            output_channels: Number of output channels.
            kernel_shape: The shape of the kernel. Either an integer or a sequence of
                length ``num_spatial_dims``.
            stride: Optional stride for the kernel. Either an integer or a sequence of
                length ``num_spatial_dims``. Defaults to 1.
            rate: Optional kernel dilation rate. Either an integer or a sequence of
                length ``num_spatial_dims``. 1 corresponds to standard ND convolution,
                ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
            padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or a
                sequence of n ``(low, high)`` integer pairs that give the padding to
                apply before and after each spatial dimension. or a callable or sequence
                of callables of size ``num_spatial_dims``. Any callables must take a
                single integer argument equal to the effective kernel size and return a
                sequence of two integers representing the padding before and after. See
                ``haiku.pad.*`` for more details and example functions. Defaults to
                ``SAME``. See:
                https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
            with_bias: Whether to add a bias. By default, true.
            w_init: Optional weight initialization. By default, truncated normal.
            b_init: Optional bias initialization. By default, zeros.
            data_format: The data format of the input.  Can be either
                ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
                default, ``channels_last``. See :func:`get_channel_index`.
            mask: Optional mask of the weights.
            feature_group_count: Optional number of groups in group convolution.
                Default value of 1 corresponds to normal dense convolution. If a higher
                value is used, convolutions are applied separately to that many groups,
                then stacked together. This reduces the number of parameters
                and possibly the compute for a given ``output_channels``. See:
                https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
            name: The name of the module.
        """
        super().__init__(name=name)
        if num_spatial_dims <= 0:
            raise ValueError(
                "We only support convolution operations for `num_spatial_dims` "
                f"greater than 0, received num_spatial_dims={num_spatial_dims}."
            )

        self.num_spatial_dims = num_spatial_dims
        self.output_channels = output_channels
        self.kernel_shape = hk_utils.replicate(
            kernel_shape, num_spatial_dims, "kernel_shape"
        )
        self.with_bias = with_bias
        self.stride = hk_utils.replicate(stride, num_spatial_dims, "strides")
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        self.mask = mask
        self.feature_group_count = feature_group_count
        self.lhs_dilation = hk_utils.replicate(1, num_spatial_dims, "lhs_dilation")
        self.kernel_dilation = hk_utils.replicate(
            rate, num_spatial_dims, "kernel_dilation"
        )
        self.data_format = data_format
        self.channel_index = hk_utils.get_channel_index(data_format)
        self.dimension_numbers = to_dimension_numbers(
            num_spatial_dims, channels_last=(self.channel_index == -1), transpose=False
        )
        self.dimension_numbers = jax.lax.ConvDimensionNumbers(
            lhs_spec=self.dimension_numbers.lhs_spec,
            rhs_spec=tuple(range(2 + num_spatial_dims)),
            out_spec=self.dimension_numbers.out_spec,
        )

        if isinstance(padding, str):
            self.padding = padding.upper()
        elif hk.pad.is_padfn(padding):
            self.padding = hk.pad.create_from_padfn(
                padding=padding,
                kernel=self.kernel_shape,
                rate=self.kernel_dilation,
                n=self.num_spatial_dims,
            )
        else:
            self.padding = hk.pad.create_from_tuple(padding, self.num_spatial_dims)

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        precision: Optional[jax.lax.Precision] = None,
    ) -> jnp.ndarray:
        """Connects ``ConvND`` layer.
        Args:
            inputs: An array of shape ``[spatial_dims, C]`` and rank-N+1 if unbatched,
                or an array of shape ``[N, spatial_dims, C]`` and rank-N+2 if batched.
            precision: Optional :class:`jax.lax.Precision` to pass to
                :func:`jax.lax.conv_general_dilated`.
        Returns:
            An array of shape ``[spatial_dims, output_channels]`` and rank-N+1 if
                unbatched, or an array of shape ``[N, spatial_dims, output_channels]``
                and rank-N+2 if batched.
        """
        unbatched_rank = self.num_spatial_dims + 1
        allowed_ranks = [unbatched_rank, unbatched_rank + 1]
        if inputs.ndim not in allowed_ranks:
            raise ValueError(
                f"Input to ConvND needs to have rank in {allowed_ranks},"
                f" but input has shape {inputs.shape}."
            )

        unbatched = inputs.ndim == unbatched_rank
        if unbatched:
            inputs = jnp.expand_dims(inputs, axis=0)

        if inputs.shape[self.channel_index] % self.feature_group_count != 0:
            raise ValueError(
                f"Inputs channels {inputs.shape[self.channel_index]} "
                f"should be a multiple of feature_group_count "
                f"{self.feature_group_count}"
            )
        w_shape = (
            self.output_channels,
            inputs.shape[self.channel_index] // self.feature_group_count,
        ) + self.kernel_shape

        if self.mask is not None and self.mask.shape != w_shape:
            raise ValueError(
                "Mask needs to have the same shape as weights. "
                f"Shapes are: {self.mask.shape}, {w_shape}"
            )

        w_init = self.w_init
        if w_init is None:
            fan_in_shape = np.prod(w_shape[:-1])
            stddev = 1.0 / np.sqrt(fan_in_shape)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)

        if self.mask is not None:
            w *= self.mask

        out = jax.lax.conv_general_dilated(
            inputs,
            w,
            window_strides=self.stride,
            padding=self.padding,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=self.dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=precision,
        )

        if self.with_bias:
            b = hk.get_parameter(
                "b", (self.output_channels,), inputs.dtype, init=self.b_init
            )
            if self.channel_index != -1:
                b = b.reshape(b.shape + (1,) * self.num_spatial_dims)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        if unbatched:
            out = jnp.squeeze(out, axis=0)
        return out


class Conv1D(ConvND):
    def __init__(self, *args, data_format="NCW", **kwargs):
        super().__init__(1, *args, data_format=data_format, **kwargs)


class Conv2D(ConvND):
    def __init__(self, *args, data_format="NCHW", **kwargs):
        super().__init__(2, *args, data_format=data_format, **kwargs)


class Conv3D(ConvND):
    def __init__(self, *args, data_format="NCDHW", **kwargs):
        super().__init__(3, *args, data_format=data_format, **kwargs)


def apply_conv_channel_parallel(conv: ConvND, x: jnp.DeviceArray) -> jnp.DeviceArray:
    return jnp.concatenate(
        jax.lax.all_gather(conv(x), "diffusion_shard"), axis=-conv.num_spatial_dims - 1
    )


def apply_upsampling_by_2(x):
    assert x.ndim == 4  # (N, C, H, W)
    return jax.image.resize(
        x, x.shape[:2] + (x.shape[2] * 2, x.shape[3] * 2), "nearest", antialias=False
    )


def apply_downsampling_by_2(x):
    assert x.ndim == 4  # (N, C, H, W)
    return hk.avg_pool(x, 2, 2, "VALID", channel_axis=-3)


def geglu(x):
    out, gate = jnp.split(x, 2, axis=-1)
    return out * jax.nn.gelu(gate, approximate=False)


class UNetTimeEmbeddingShard(hk.Module):
    def __init__(self, config, activation="silu", name=None):
        super().__init__(name=name)

        self.embed_dim = config["block_out_channels"][0]
        self.shards = config["cores_per_replica"]
        assert self.embed_dim % self.shards == 0
        self.embed_dim_per_shard = self.embed_dim // self.shards
        assert self.embed_dim % 2 == 0
        assert self.embed_dim % self.shards == 0

        self.inv_freq = 1.0 / (
            10000 ** (np.arange(0, self.embed_dim, 2) / self.embed_dim)
        )

        self.dense_proj = hk.Linear(4 * self.embed_dim_per_shard, name="dense_proj")
        self.activation_fn = getactfn(activation)
        self.dense_proj_o = hk.Linear(4 * self.embed_dim, name="dense_proj_o")

    def __call__(self, t: Union[float, int, jnp.DeviceArray], channels_out: int):
        if isinstance(t, float) or isinstance(t, int):
            t: jnp.DeviceArray = jnp.array(t, dtype=jnp.float32)
        assert t.ndim == 0
        t = jnp.broadcast_to(t, channels_out)

        time_embeddings = jnp.outer(t, self.inv_freq)
        time_embeddings = jnp.concatenate(
            (jnp.sin(time_embeddings), jnp.cos(time_embeddings)), axis=-1
        )
        time_embeddings = self.dense_proj(time_embeddings)
        time_embeddings = self.activation_fn(time_embeddings)
        time_embeddings = jax.lax.psum(
            self.dense_proj_o(time_embeddings), "diffusion_shard"
        )

        return time_embeddings


class ResNetShard(hk.Module):
    def __init__(
        self,
        shards: int,
        channels_in: int,
        channels_out: int,
        groups_in: Optional[int] = None,
        groups_out: int = 32,
        time_embedding_channels: Optional[int] = None,
        upsample: bool = False,
        downsample: bool = False,
        activation: str = "silu",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.shards = shards
        self.upsample = upsample
        self.downsample = downsample
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.groups_out = groups_out
        self.groups_in = groups_in if groups_in is not None else self.groups_out
        self.time_embedding_channels = time_embedding_channels

        assert self.channels_out % self.shards == 0
        self.channels_out_per_shard = self.channels_out // self.shards

        self.activation_fn = getactfn(activation)

        self.norm_1 = ReplicatedGroupNorm(self.groups_in, name="norm_1")
        self.conv_1 = Conv2D(
            self.channels_out_per_shard, kernel_shape=3, padding=(1, 1), name="conv_1"
        )
        self.t_proj = (
            hk.Linear(self.channels_out_per_shard, name="t_proj")
            if self.time_embedding_channels is not None
            else None
        )
        self.norm_2 = ReplicatedGroupNorm(self.groups_out, name="norm_2")
        self.conv_2 = Conv2D(
            self.channels_out_per_shard, kernel_shape=3, padding=(1, 1), name="conv_2"
        )
        self.conv_shortcut = (
            Conv2D(
                self.channels_out_per_shard,
                kernel_shape=1,
                padding=(0, 0),
                name="conv_shortcut",
            )
            if self.channels_in != self.channels_out
            else None
        )

    def __call__(self, x, time_embeddings=None):
        assert x.shape[-3] == self.channels_in
        residual = x
        x = self.norm_1(x)
        x = self.activation_fn(x)
        if self.downsample:
            x = apply_downsampling_by_2(x)
            residual = apply_downsampling_by_2(residual)
        if self.upsample:
            x = apply_upsampling_by_2(x)
            residual = apply_upsampling_by_2(residual)
        x = apply_conv_channel_parallel(self.conv_1, x)
        if time_embeddings is not None:
            time_embeddings = self.activation_fn(time_embeddings)
            time_embeddings = jnp.concatenate(
                jax.lax.all_gather(self.t_proj(time_embeddings), "diffusion_shard"),
                axis=-1,
            )
            x = x + time_embeddings.reshape(time_embeddings.shape + (1, 1))
        x = self.norm_2(x)
        x = self.activation_fn(x)
        x = apply_conv_channel_parallel(self.conv_2, x)
        if self.channels_in != self.channels_out:
            residual = apply_conv_channel_parallel(self.conv_shortcut, residual)
        return x + residual


class UNetAttentionShard(hk.Module):
    def __init__(self, shards, channels_out, heads, has_qkv_bias=False, name=None):
        super().__init__(name=name)
        self.heads = heads
        self.dim = channels_out
        self.shards = shards
        assert self.dim % self.heads == 0
        assert self.dim % self.shards == 0
        assert self.heads % self.shards == 0
        self.dim_per_head = self.dim // self.heads
        self.dim_per_shard = self.dim // self.shards
        self.heads_per_shard = self.heads // self.shards
        self.use_combined_qkv = False

        self.q = hk.Linear(self.dim_per_shard, with_bias=has_qkv_bias, name="q")
        self.v = hk.Linear(self.dim_per_shard, with_bias=has_qkv_bias, name="v")
        self.k = hk.Linear(self.dim_per_shard, with_bias=has_qkv_bias, name="k")
        self.o = hk.Linear(self.dim, with_bias=True, name="o")

    def qvk_proj(self, x, y):
        q = self.q(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
        v = self.v(y).reshape(y.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
        k = self.k(y).reshape(y.shape[:-1] + (self.heads_per_shard, self.dim_per_head))

        return q, v, k

    def cross_attn(self, q, v, k, attn_bias):
        attention_logits = jnp.einsum("bthd,bThd->bhtT", q, k)

        sqrt_key_size = np.sqrt(self.dim_per_head).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size

        attention_logits += attn_bias

        attention_weights = jax.nn.softmax(attention_logits)
        attention_vec = jnp.einsum("bhtT,bThd->bthd", attention_weights, v).reshape(
            (q.shape[0], -1, self.dim_per_shard)
        )

        return self.o(attention_vec)

    def __call__(self, x, y=None):
        if y is None:
            y = x
        q, v, k = self.qvk_proj(x, y)
        x = self.cross_attn(q, v, k, 0)
        x = jax.lax.psum(x, "diffusion_shard")
        return x


class UNetLayerShard(hk.Module):
    def __init__(
        self,
        shards,
        heads,
        channels_in,
        channels_out,
        time_embedding_channels: Optional[int] = None,
        activation="silu",
        has_attention=False,
        attention_first=False,
        has_qkv_bias=False,
        name=None,
    ):
        super().__init__(name=name)
        self.time_embedding_channels = time_embedding_channels
        self.heads = heads
        self.dim = channels_out
        self.shards = shards
        assert self.dim % self.heads == 0
        assert self.dim % self.shards == 0
        self.dim_per_head = self.dim // self.heads
        self.dim_per_shard = self.dim // self.shards
        self.has_attention = has_attention
        self.attention_first = attention_first

        self.resnet = ResNetShard(
            shards=self.shards,
            channels_in=channels_in,
            channels_out=channels_out,
            groups_out=32,
            time_embedding_channels=self.time_embedding_channels,
            activation=activation,
            name="resnet",
        )
        if has_attention:
            self.norm_in = ReplicatedGroupNorm(32, name="norm_in")
            self.conv_in = Conv2D(
                self.dim_per_shard, kernel_shape=1, padding=(0, 0), name="conv_in"
            )
            self.norm_1 = ReplicatedLayerNorm(name="norm_1")
            self.attn_1 = UNetAttentionShard(
                self.shards,
                self.dim,
                self.heads,
                has_qkv_bias=has_qkv_bias,
                name="attn_1",
            )
            self.norm_2 = ReplicatedLayerNorm(name="norm_2")
            self.attn_2 = UNetAttentionShard(
                self.shards,
                self.dim,
                self.heads,
                has_qkv_bias=has_qkv_bias,
                name="attn_2",
            )
            self.post_attn_norm = ReplicatedLayerNorm(name="post_attn_norm")
            self.dense_proj = hk.Linear(8 * self.dim_per_shard, name="dense_proj")
            self.dense_proj_o = hk.Linear(self.dim, name="dense_proj_o")
            self.conv_out = Conv2D(
                self.dim_per_shard, kernel_shape=1, padding=(0, 0), name="conv_out"
            )

    def ff(self, x):
        x = self.dense_proj(x)
        x = geglu(x)
        return self.dense_proj_o(x)

    def __call__(self, x, y=None, time_embeddings=None):
        if not self.attention_first:
            x = self.resnet(x, time_embeddings=time_embeddings)

        if self.has_attention:
            if y is None:
                y = x

            residual = x
            N, C, H, W = x.shape
            x = self.norm_in(x)
            x = apply_conv_channel_parallel(self.conv_in, x)
            x = x.reshape((N, C, H * W)).transpose((0, 2, 1))

            x = x + self.attn_1(self.norm_1(x))
            x = x + self.attn_2(self.norm_2(x), y)
            x = x + jax.lax.psum(self.ff(self.post_attn_norm(x)), "diffusion_shard")

            x = x.transpose((0, 2, 1)).reshape((N, C, H, W))
            x = apply_conv_channel_parallel(self.conv_out, x)
            x = x + residual

        if self.attention_first:
            x = self.resnet(x, time_embeddings=time_embeddings)

        return x


class UNetSimpleLayerShard(hk.Module):
    def __init__(self, config, *args, name=None, **kwargs):
        super().__init__(name=name)
        self.config = config
        self.norm_in = ReplicatedGroupNorm(32, name="norm_in")
        self.attn_1 = UNetAttentionShard(*args, **kwargs, name="attn_1")

    def __call__(self, x, y=None):
        residual = x
        N, C, H, W = x.shape
        x = self.norm_in(x)
        x = x.reshape((N, C, H * W)).transpose((0, 2, 1))
        x = self.attn_1(x, y=y)
        x = x.transpose((0, 2, 1)).reshape((N, C, H, W))
        x = x + residual
        return x


class UNetDownShard(hk.Module):
    def __init__(
        self,
        config,
        layers,
        channels_in,
        channels_out,
        time_embedding_channels: Optional[int] = None,
        heads: Optional[int] = None,
        activation="silu",
        has_attention=False,
        has_convolution=True,
        has_qkv_bias=False,
        name=None,
    ):
        super().__init__(name=name)
        self.config = config
        self.shards = config["cores_per_replica"]
        self.channels_out = channels_out
        self.n_layers = layers
        self.has_attention = has_attention
        self.has_convolution = has_convolution
        self.time_embedding_channels = time_embedding_channels
        self.heads = heads

        assert self.channels_out % self.shards == 0
        self.channels_out_per_shard = self.channels_out // self.shards

        self.layers = [
            UNetLayerShard(
                self.shards,
                channels_in=channels_in if i == 0 else self.channels_out,
                channels_out=self.channels_out,
                heads=self.heads,
                has_attention=self.has_attention,
                time_embedding_channels=self.time_embedding_channels,
                has_qkv_bias=has_qkv_bias,
                activation=activation,
                name=f"layer_{i}",
            )
            for i in range(self.n_layers)
        ]

        self.conv = (
            Conv2D(
                self.channels_out_per_shard,
                kernel_shape=3,
                padding=(1, 1),
                stride=2,
                name="conv",
            )
            if self.has_convolution
            else None
        )

    def __call__(self, x, y=None, time_embeddings=None):
        residuals = []
        for layer in self.layers:
            x = layer(x, y=y, time_embeddings=time_embeddings)
            residuals.append(x)
        if self.has_convolution:
            x = apply_conv_channel_parallel(self.conv, x)
            residuals.append(x)
        return residuals


class UNetMiddleShard(hk.Module):
    def __init__(
        self,
        config,
        channels_out,
        time_embedding_channels: Optional[int] = None,
        heads: Optional[int] = None,
        simple=False,
        has_qkv_bias=False,
        activation="silu",
        name=None,
    ):
        super().__init__(name=name)
        self.config = config
        self.shards = config["cores_per_replica"]
        self.time_embedding_channels = time_embedding_channels
        self.channels_out = channels_out
        self.heads = heads

        self.resnet = ResNetShard(
            shards=self.shards,
            channels_in=self.channels_out,
            channels_out=self.channels_out,
            groups_out=32,
            time_embedding_channels=self.time_embedding_channels,
            activation=activation,
            name="resnet",
        )
        if simple:
            self.layer = UNetSimpleLayerShard(
                config,
                self.shards,
                channels_out=self.channels_out,
                heads=self.heads,
                has_qkv_bias=has_qkv_bias,
                name="layer_0",
            )
        else:
            self.layer = UNetLayerShard(
                self.shards,
                channels_in=self.channels_out,
                channels_out=self.channels_out,
                heads=self.heads,
                has_attention=True,
                attention_first=True,
                time_embedding_channels=self.time_embedding_channels,
                has_qkv_bias=has_qkv_bias,
                activation=activation,
                name="layer_0",
            )

    def __call__(self, x, y=None, time_embeddings=None):
        x = self.resnet(x, time_embeddings=time_embeddings)
        x = self.layer(x, y=y, time_embeddings=time_embeddings)
        return x


class UNetUpShard(hk.Module):
    def __init__(
        self,
        config,
        layers,
        channels_in,
        channels_out,
        last_layer_residual_channels_in: Optional[int] = None,
        time_embedding_channels: Optional[int] = None,
        heads: Optional[int] = None,
        activation="silu",
        has_attention=False,
        has_convolution=True,
        has_qkv_bias=False,
        name=None,
    ):
        super().__init__(name=name)
        self.config = config
        self.shards = config["cores_per_replica"]
        self.channels_out = channels_out
        self.n_layers = layers
        self.has_attention = has_attention
        self.has_convolution = has_convolution
        self.time_embedding_channels = time_embedding_channels
        self.heads = heads

        assert self.channels_out % self.shards == 0
        self.channels_out_per_shard = self.channels_out // self.shards

        self.layers = [
            UNetLayerShard(
                self.shards,
                channels_in=(channels_in if i == 0 else channels_out)
                + (
                    0
                    if last_layer_residual_channels_in is None
                    else last_layer_residual_channels_in
                    if i == self.n_layers - 1
                    else channels_out
                ),
                channels_out=self.channels_out,
                has_attention=self.has_attention,
                heads=self.heads,
                time_embedding_channels=self.time_embedding_channels,
                has_qkv_bias=has_qkv_bias,
                activation=activation,
                name=f"layer_{i}",
            )
            for i in range(self.n_layers)
        ]

        self.conv = (
            Conv2D(
                self.channels_out_per_shard, kernel_shape=3, padding=(1, 1), name="conv"
            )
            if self.has_convolution
            else None
        )

    def __call__(
        self, x, y=None, down_residuals: Optional[list] = None, time_embeddings=None
    ):
        if down_residuals is not None:
            for layer, down_residual in zip(self.layers, reversed(down_residuals)):
                x = jnp.concatenate((x, down_residual), axis=-3)
                x = layer(x, y=y, time_embeddings=time_embeddings)
        else:
            for layer in self.layers:
                x = layer(x, y=y, time_embeddings=time_embeddings)
        if self.has_convolution:
            x = apply_upsampling_by_2(x)
            x = apply_conv_channel_parallel(self.conv, x)
        return x


class UNetOutShard(hk.Module):
    def __init__(self, config, out_channels, activation="silu", name=None):
        super().__init__(name=name)
        self.config = config

        self.norm = ReplicatedGroupNorm(32, name="norm")
        self.activation_fn = getactfn(activation)
        self.conv = Conv2D(out_channels, kernel_shape=3, padding=(1, 1), name="conv")

    def __call__(self, x):
        x = self.norm(x)
        x = self.activation_fn(x)
        return self.conv(x)
