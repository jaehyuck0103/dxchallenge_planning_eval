"""Policy network for the V-Max SAC baseline (LQ encoder + MLP head).

Verbatim port of the V-Max modules this checkpoint was trained with
(vmax/agents/networks/encoders/{lq,attention_utils,embedding_utils}.py,
decoders/mlp.py, network_factory.py). Module attribute names, explicit
``name=`` arguments and layer creation order are preserved exactly so the
flax parameter tree matches the exported weights.
"""

from collections.abc import Callable, Sequence
from functools import partial

import einops
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from flax import linen as nn


def default(val, d):
    """Return val if val is not None, else d."""
    return val if val is not None else d


class FeedForward(nn.Module):
    """Feed forward network with GELU activation and dropout."""

    mult: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        features = x.shape[-1]

        x = nn.Dense(features * self.mult)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(features)(x)

        return x


class ReZero(nn.Module):
    """ReZero block which scales the output."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param("scale", init.zeros, (1,))

        return scale * x


class AttentionLayer(nn.Module):
    """Attention layer that computes self or cross attention."""

    heads: int = 8
    head_features: int = 64
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self, x: jax.Array, context=None, mask_k=None, mask_q=None, deterministic: bool = False
    ) -> jax.Array:
        # mask is on context(k)
        h = self.heads
        dim = self.head_features * h

        q = nn.Dense(dim, use_bias=False)(x)
        k = nn.Dense(dim, use_bias=False)(default(context, x))
        v = nn.Dense(dim, use_bias=False)(default(context, x))

        q, k, v = map(lambda arr: einops.rearrange(arr, "b n (h d) -> b n h d", h=h), (q, k, v))
        sim = jnp.einsum("b i h d, b j h d -> b i j h", q, k) * self.head_features**-0.5

        if mask_k is not None:
            big_neg = jnp.finfo(jnp.float32).min
            sim = jnp.where(mask_k[:, None, :, None], sim, big_neg)
        if mask_q is not None:
            big_neg = jnp.finfo(jnp.float32).min
            sim = jnp.where(mask_q[:, :, None, None], sim, big_neg)

        attn = nn.softmax(sim, axis=-2)
        out = jnp.einsum("b i j h, b j h d -> b i h d", attn, v)
        out = einops.rearrange(out, "b n h d -> b n (h d)", h=h)

        out = nn.Dense(x.shape[-1])(out)
        out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

        return out


def build_mlp_embedding(input_features, output_size, hidden_sizes, activation_fn, name_prefix):
    """Build an MLP embedding network."""
    x = input_features
    for i, hidden_size in enumerate(hidden_sizes):
        x = nn.Dense(hidden_size, name=f"{name_prefix}_layer_{i}")(x)
        if activation_fn:
            x = activation_fn(x)

    output = nn.Dense(output_size, name=f"{name_prefix}_output")(x)

    return output


class LQAttention(nn.Module):
    """Latent-Query attention module."""

    depth: int = 4
    num_latents: int = 64
    latent_num_heads: int = 2
    latent_head_features: int = 64
    cross_num_heads: int = 2
    cross_head_features: int = 64
    ff_mult: int = 4
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    tie_layer_weights: bool = False

    @nn.compact
    def __call__(self, x, mask=None):
        bs, dim = x.shape[0], x.shape[-1]

        # Learnable latent feature
        latents = self.param("latents", init.normal(), (self.num_latents, dim * self.ff_mult))
        latent = einops.repeat(latents, "n d -> b n d", b=bs)

        # Cross, self attention and feedforward layers
        cross_attn = partial(
            AttentionLayer,
            heads=self.cross_num_heads,
            head_features=self.cross_head_features,
            dropout=self.attn_dropout,
        )
        self_attn = partial(
            AttentionLayer,
            heads=self.latent_num_heads,
            head_features=self.latent_head_features,
            dropout=self.attn_dropout,
        )
        ff = partial(FeedForward, mult=self.ff_mult, dropout=self.ff_dropout)

        # weights optionnaly shared between repeats - Page 2 paper Perceiver
        if self.tie_layer_weights:
            ca = cross_attn(name="cross_attn")
            sa = self_attn(name="self_attn")
            cf = ff(name="cross_ff")
            lf = ff(name="self_ff")
            for i in range(self.depth):
                rz = ReZero(name=f"rezero_cross_{i}")
                latent += rz(ca(latent, x, mask_k=mask))
                latent += rz(cf(latent))
                rz = ReZero(name=f"rezero_self_{i}")
                latent += rz(sa(latent))
                latent += rz(lf(latent))
        else:
            for i in range(self.depth):
                rz = ReZero(name=f"rezero_cross{i}")
                latent += rz(cross_attn(name=f"cross_attn_{i}")(latent, x, mask_k=mask))
                latent += rz(ff(name=f"cross_ff_{i}")(latent))
                rz = ReZero(name=f"rezero_self_{i}")
                latent += rz(self_attn(name=f"latent_attn_{i}")(latent))
                latent += rz(ff(name=f"latent_ff_{i}")(latent))

        return latent


class LQEncoder(nn.Module):
    """Latent-Query encoder module."""

    unflatten_fn: Callable = lambda x: x
    embedding_layer_sizes: tuple[int] = (256, 256)
    embedding_activation: Callable = nn.relu
    encoder_depth: int = 4
    dk: int = 64
    num_latents: int = 64
    latent_num_heads: int = 2
    latent_head_features: int = 64
    cross_num_heads: int = 2
    cross_head_features: int = 64
    ff_mult: int = 4
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    tie_layer_weights: bool = False

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        # Get features and masks
        features, masks = self.unflatten_fn(obs)
        sdc_traj_features, other_traj_features, rg_features, tl_features, gps_path_features = (
            features
        )
        sdc_traj_valid_mask, other_traj_valid_mask, rg_valid_mask, tl_valid_mask = masks

        # Embeddings for all sub features
        num_objects, timestep_agent = other_traj_features.shape[-3:-1]
        num_roadgraph = rg_features.shape[-2]
        target_len = gps_path_features.shape[-2]
        num_light, timestep_tl = tl_features.shape[-3:-1]

        # Latent encoding
        sdc_traj_encoding = build_mlp_embedding(
            sdc_traj_features,
            self.dk,
            self.embedding_layer_sizes,
            self.embedding_activation,
            "sdc_traj_enc",
        )
        other_traj_encoding = build_mlp_embedding(
            other_traj_features,
            self.dk,
            self.embedding_layer_sizes,
            self.embedding_activation,
            "other_traj_enc",
        )
        rg_encoding = build_mlp_embedding(
            rg_features,
            self.dk,
            self.embedding_layer_sizes,
            self.embedding_activation,
            "rg_enc",
        )
        tl_encoding = build_mlp_embedding(
            tl_features,
            self.dk,
            self.embedding_layer_sizes,
            self.embedding_activation,
            "tl_enc",
        )
        gps_path_encoding = build_mlp_embedding(
            gps_path_features,
            self.dk,
            self.embedding_layer_sizes,
            self.embedding_activation,
            "gps_path_enc",
        )

        # Positional Encoding
        sdc_traj_encoding += jnp.expand_dims(
            self.param("sdc_traj_pe", init.normal(), (1, timestep_agent, self.dk)), 0
        )
        other_traj_encoding += jnp.expand_dims(
            self.param("other_traj_pe", init.normal(), (num_objects, timestep_agent, self.dk)),
            0,
        )
        rg_encoding += jnp.expand_dims(
            self.param("rg_pe", init.normal(), (num_roadgraph, self.dk)), 0
        )
        tl_encoding += jnp.expand_dims(
            self.param("tj_pe", init.normal(), (num_light, timestep_tl, self.dk)), 0
        )
        gps_path_encoding += jnp.expand_dims(
            self.param("gps_path_pe", init.normal(), (target_len, self.dk)), 0
        )

        # Flatten by NumAgent NumObsTS , Feature_dim
        sdc_traj_encoding = einops.rearrange(sdc_traj_encoding, "b n t d -> b (n t) d")
        other_traj_encoding = einops.rearrange(other_traj_encoding, "b n t d -> b (n t) d")
        tl_encoding = einops.rearrange(tl_encoding, "b n t d -> b (n t) d")

        # Masks
        sdc_traj_valid_mask = einops.rearrange(sdc_traj_valid_mask, "b n t -> b (n t)")
        other_traj_valid_mask = einops.rearrange(other_traj_valid_mask, "b n t -> b (n t)")
        tl_valid_mask = einops.rearrange(tl_valid_mask, "b n t -> b (n t)")
        gps_path_mask = jnp.ones(gps_path_encoding.shape[:-1])

        # [B, N, D]
        input = jnp.concatenate(
            [sdc_traj_encoding, other_traj_encoding, rg_encoding, tl_encoding, gps_path_encoding],
            axis=1,
        )
        mask = jnp.concatenate(
            [
                sdc_traj_valid_mask,
                other_traj_valid_mask,
                rg_valid_mask,
                tl_valid_mask,
                gps_path_mask,
            ],
            axis=1,
        )  # [B, N]

        output = LQAttention(
            depth=self.encoder_depth,
            num_latents=self.num_latents,
            latent_num_heads=self.latent_num_heads,
            latent_head_features=self.latent_head_features,
            cross_num_heads=self.cross_num_heads,
            cross_head_features=self.cross_head_features,
            ff_mult=self.ff_mult,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            tie_layer_weights=self.tie_layer_weights,
            name="lq_attention",
        )(input, mask)

        output = output.mean(axis=1)

        return output


class MLP(nn.Module):
    """Multi-layer perceptron network composed of dense layers."""

    layer_sizes: Sequence[int] = (256, 256)
    activation: Callable = nn.relu
    dropout_rate: float | None = None
    kernel_init: Callable = nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        for i, size in enumerate(self.layer_sizes):
            x = nn.Dense(size, kernel_init=self.kernel_init, name=f"hidden_{i}")(x)

            if i != len(self.layer_sizes) - 1:
                x = self.activation(x)
            if self.dropout_rate is not None:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        return x


class PolicyNetwork(nn.Module):
    """Policy network module that builds the forward propagation path."""

    encoder_layer: nn.Module | None = None
    fully_connected_layer: nn.Module | None = None
    final_activation: Callable | None = None

    output_size: int = 1

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = self.encoder_layer(obs) if self.encoder_layer is not None else obs
        x = self.fully_connected_layer(x)
        x = nn.Dense(self.output_size)(x)

        if self.final_activation:
            x = self.final_activation(x)

        return x


def build_policy_network(unflatten_fn: Callable) -> PolicyNetwork:
    """Assemble the policy network with this run's architecture.

    Mirrors V-Max ``make_policy_network`` with the resolved hydra config of
    run 260623_1_rideflux: LQ encoder (network.encoder) + MLP head
    (algorithm.network.policy) + Dense(4) for the gaussian (loc, scale) params.
    """
    encoder = LQEncoder(
        unflatten_fn=unflatten_fn,
        embedding_layer_sizes=(256, 256),
        embedding_activation=nn.relu,
        encoder_depth=4,
        dk=64,
        num_latents=16,
        latent_num_heads=2,
        latent_head_features=16,
        cross_num_heads=2,
        cross_head_features=16,
        ff_mult=2,
        attn_dropout=0.0,
        ff_dropout=0.0,
        tie_layer_weights=True,
    )
    fully_connected = MLP(layer_sizes=(256, 64, 32), activation=nn.relu)

    return PolicyNetwork(
        encoder_layer=encoder,
        fully_connected_layer=fully_connected,
        final_activation=None,
        output_size=4,  # NormalTanhDistribution params: 2 * action_size
    )


def deterministic_action(logits: jax.Array) -> jax.Array:
    """SAC deterministic action = mode of the tanh-squashed gaussian = tanh(loc)."""
    loc, _ = jnp.split(logits, 2, axis=-1)
    return jnp.tanh(loc)
