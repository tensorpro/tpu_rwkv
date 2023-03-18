import math
from dataclasses import dataclass
from typing import Any, List, Type, Union

import jax
import jax.numpy as jnp
from flax import linen as nn

@dataclass
class Config:
    num_layers: int = 2
    embedding_size: int = 2
    pos_embedding_size: int = 0
    vocab_size: int = 3
    context_length: int = 3
    attention_at_layer: int = 3
    attention_size: int = 3
    head_qk_size: int = 0
    grad_cp: bool = False
    channel_mix_at_input: bool = False
    dtype: str = 'bfloat16'


def wkv_single_channel(wc, uc, kc, vc, state_in_c=(0,0,-jnp.inf)):
    """
    credit for this kernel: github.com/BlealTan
    """

    def step(pqo, kct_vct):
        kct, vct = kct_vct

        p, q, o = pqo
        no = jnp.maximum(o, uc + kct)
        A = jnp.exp(o - no)
        B = jnp.exp(uc + kct - no)
        y = (A * p + B * vct) / (A * q + B)

        no = jnp.maximum(wc + o, kct)
        A = jnp.exp(wc + o - no)
        B = jnp.exp(kct - no)
        p = A * p + B * vct
        q = A * q + B
        o = no

        return (p, q, o), y

    state_out_c, y = jax.lax.scan(step, state_in_c, (kc, vc))
    return y, state_out_c

@jax.jit
def wkv(w,u,k,v):
    w = -jnp.exp(w)
    return jax.vmap(wkv_single_channel, -1, -1)(w,u,k,v)[0]

@jax.jit
def time_shift(x):
    return jnp.pad(x, [(1, 0), (0, 0)])[:-1, :]


def initialize_to_value(x, dtype):
    """
    makes an initializer function that ignores the given PRNGKey
    and always returns the given value
    """
    return lambda _: x.astype(dtype)

class TimeMix(nn.Module):
    config: Config
    layer_depth: int

    def setup(self):
        """
        Initializes a time mix module for a given config / layer depth.
        """
        config = self.config
        layer_depth = self.layer_depth
        embedding_size = config.embedding_size
        num_layers = config.num_layers

        # goes from 0 to 1 along layer depth
        ratio_0_to_1 = layer_depth / (num_layers - 1)
        # goes from 1 to (almost) 0 along layer depth
        ratio_1_to_almost_0 = 1.0 - (layer_depth / num_layers)
        zigzag = .5 * (jnp.arange(1, embedding_size+1) % 3 - 1)
        time_first = jnp.full(embedding_size, math.log(.3)) + zigzag
        self.time_first = self.param('time_first', initialize_to_value(time_first, config.dtype))

        x = (jnp.arange(embedding_size) /
             embedding_size)[jnp.newaxis, :]
        time_mix_k = jnp.power(x, ratio_1_to_almost_0)
        time_mix_v = time_mix_k + .3 * ratio_0_to_1
        time_mix_r = jnp.power(x, .5 * ratio_1_to_almost_0)

        self.time_mix_k = self.param('time_mix_k', initialize_to_value(time_mix_k, config.dtype))
        self.time_mix_v = self.param('time_mix_v', initialize_to_value(time_mix_v, config.dtype))
        self.time_mix_r = self.param('time_mix_r', initialize_to_value(time_mix_r, config.dtype))

        h = jnp.arange(0, embedding_size)
        decay_speed = -5 + 8 * (h / (embedding_size - 1)
                                ) ** (.7 + 1.3 * ratio_0_to_1)
        self.time_decay = self.param('time_decay', initialize_to_value(decay_speed, config.dtype))
        self.key = nn.Dense(embedding_size, use_bias=False, dtype=config.dtype)
        self.value = nn.Dense(embedding_size, use_bias=False, dtype=config.dtype)
        self.receptance = nn.Dense(embedding_size, use_bias=False, dtype=config.dtype)
        self.output = nn.Dense(embedding_size, use_bias=False, dtype=config.dtype)

    @nn.jit
    def __call__(self, x):
        T, C = x.shape

        # performs time_shift.
        xx = time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = jax.nn.sigmoid(r)
        rwkv = sr * wkv(self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        return rwkv


class ChannelMix(nn.Module):
    config: Config
    layer_depth: int

    def setup(self):
        layer_number = self.layer_depth
        config = self.config
        ratio_1_to_almost0 = 1.0 - (layer_number / config.num_layers)

        embedding_size = config.embedding_size
        x = (jnp.arange(embedding_size) /
             embedding_size)[jnp.newaxis, ...]

        self.time_mix_k = jnp.power(x, ratio_1_to_almost0)
        self.time_mix_r = jnp.power(x, ratio_1_to_almost0)

        hidden_size = 4 * embedding_size
        self.key = nn.Dense(hidden_size, use_bias=False, dtype=config.dtype)
        self.receptance = nn.Dense(embedding_size, use_bias=False, dtype=config.dtype)
        self.value = nn.Dense(embedding_size, use_bias=False, dtype=config.dtype)

    @nn.jit
    def __call__(self, x):
        xx = time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = jnp.square(jax.nn.relu(k))
        kv = self.value(k)
        rkv = jax.nn.sigmoid(self.receptance(xr)) * kv
        return rkv


class Block(nn.Module):
    config: Config
    layer_num: int
    mix1_type: Union[Type[ChannelMix], Type[TimeMix]] = TimeMix

    def setup(self):
        config, layer_num = self.config, self.layer_num
        embedding_size = config.embedding_size
        position_embedding_size = config.pos_embedding_size
        self.layer_norm1 = nn.LayerNorm(epsilon=1e-5, dtype=config.dtype)
        self.layer_norm2 = nn.LayerNorm(epsilon=1e-5, dtype=config.dtype)
        self.mix1 = self.mix1_type(config, layer_num)
        self.mix2 = ChannelMix(config, layer_num)

    @nn.jit
    def __call__(self, x):
        T, C = x.shape
        x = x + self.mix1(self.layer_norm1(x))
        x = x + self.mix2(self.layer_norm2(x))
        return x


class TinyAttention(nn.Module):
    config: Config

    def setup(self):
        config = self.config
        embedding_size, attention_size = config.embedding_size, config.attention_size
        self.layernorm = nn.LayerNorm(epsilon=1e-5, dtype=config.dtype)
        self.mask = jnp.tril(
            jnp.ones((self.config.context_length, self.config.context_length))) == 1
        self.q = nn.Dense(attention_size, use_bias=False, dtype=config.dtype)
        self.k = nn.Dense(attention_size, use_bias=False, dtype=config.dtype)
        self.v = nn.Dense(embedding_size, use_bias=False, dtype=config.dtype)

    @nn.jit
    def __call__(self, x, x_emb):
        T, C = x.shape
        xx = self.layernorm(x)
        q = self.q(xx)[:T, :]
        k = self.k(xx)[:T, :]
        c = (q @ k.T) * (self.config.attention_size ** (-0.5))
        c = jnp.where(self.mask[:T, :T], c, 0)
        x = x + (c @ self.v(x_emb))
        return x


class PositionEmbedding(nn.Module):
    config: Config

    def setup(self):
        super().__init__()
        config = self.config
        self.pos_emb_x = self.variable("params", "pos_emb_x", jnp.zeros,
                                       (1, config.pos_embedding_size, config.embedding_size))
        self.pos_emb_y = self.variable("params", "pos_emb_y", jnp.zeros,
                                       (config.pos_embedding_size, 1, config.embedding_size))

    @nn.jit
    def __call__(self, x):
        T, C = x.shape
        pos_emb = (self.variables['params']['pos_emb_x'] +
                   self.variables['params']['pos_emb_y']).reshape(T+1, -1)[:-1, :]
        x = x + pos_emb
        return x


class HeadQK(nn.Module):
    config: Config

    def setup(self):
        config = self.config
        context_length = config.context_length
        self.head_q = nn.Dense(self.config.head_qk_size, use_bias=False, dtype=config.dtype)
        self.head_k = nn.Dense(self.config.head_qk_size, use_bias=False, dtype=config.dtype)
        self.copy_mask = jnp.tril(
            jnp.ones((context_length, context_length))) == 1

    @nn.jit
    def __call__(self, x, idx):
        T, C = x.shape
        q = self.head_q(x)[:T, :]
        k = self.head_k(x)[:T, :]
        c = (q @ k.T) * (1.0 / (self.config.head_qk_size))
        c = jnp.where(self.copy_mask[:T, :T], c, 0)
        one_hot = nn.one_hot(
            idx, num_classes=self.config.vocab_size).astype(x.dtype)
        return c @ one_hot


class RWKV(nn.Module):
    config: Config

    def setup(self):
        config = self.config
        self.emb = nn.Embed(config.vocab_size, config.embedding_size, dtype=config.dtype)
        self.ln_in = nn.LayerNorm(epsilon=1e-5, dtype=config.dtype)

        if 0 <= config.attention_at_layer < config.num_layers:
            self.tiny_attention = TinyAttention(config)

        blocks = []
        for i in range(config.num_layers):
            mix = ChannelMix if i == 0 and config.channel_mix_at_input else TimeMix
            blocks.append(Block(config, i, mix))
        self.blocks = blocks

        self.ln_out = nn.LayerNorm(epsilon=1e-5, dtype=config.dtype)
        self.head = nn.Dense(config.vocab_size, use_bias=False, dtype=config.dtype)

        # self.head_qk = lambda *_: 0.
        if config.head_qk_size > 0:
            self.head_qk = HeadQK(config)

    @nn.jit
    def __call__(self, idx):
        config = self.config
        T = len(idx)
        assert T <= config.context_length, "Cannot forward, model ctx_len is exhausted."
        x = self.emb(idx)
        x_emb = x

        x = self.ln_in(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if config.attention_at_layer == i:
                x = self.tiny_attention(x, x_emb)
        x = self.ln_out(x)

        c = 0
        if config.head_qk_size:
            c = self.head_qk(x, idx)
        x = self.head(x) + c
        return x


BatchRWKV = nn.vmap(RWKV, in_axes=0, out_axes=0,
                    variable_axes={'params': None},
                    split_rngs={'params': False})
