import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type, Union

import equinox as eqx
import flax
import jax
import jax.numpy as jnp
import numpy as np
import treeo as to
from flax import linen as nn
from jax import lax
from jaxtyping import Array, Float


@dataclass
class Config:
    num_layers: int = 2
    embedding_size: int = 2
    pos_embedding_size: int = 2
    vocab_size: int = 3
    context_length: int = 3
    attention_at_layer: int = 3
    attention_size: int = 3
    head_qk_size: int = 0
    grad_cp: bool = False
    channel_mix_at_input:bool = False


@jax.jit
def wkv(w, u, k, v):
    T, C = k.shape
    time_curve = jnp.arange(-T+2, 1)[jnp.newaxis, ...]
    k, v = map(jnp.array, [[k], [v]])
    w = -jnp.exp(w)
    ek = jnp.exp(k.transpose((0, 2, 1)))
    ekv = ek * v.transpose((0, 2, 1))
    ew_time = jnp.expand_dims(jnp.exp(w), 1) * time_curve
    time_w = jnp.concatenate([ew_time, jnp.expand_dims(u, 1)], axis=1)
    w = jnp.expand_dims(jnp.exp(time_w), 1)

    def pad(x): return jnp.pad(x, [(0, 0), (0, 0), (T-1, 0)])

    wkv = lax.conv_general_dilated(pad(ekv), w, (1,), [(
        0, 0)], dimension_numbers=('NCW', 'OIW', 'NCW'), feature_group_count=C)
    wk = lax.conv_general_dilated(pad(ek), w, (1,), [(
        0, 0)], dimension_numbers=('NCW', 'OIW', 'NCW'), feature_group_count=C)
    return (wkv / wk).transpose(0, 2, 1)[0].T

@jax.jit
def time_shift(x):
    return jnp.pad(x, [(1, 0), (0, 0)])[:-1, :]


def simple_param(module, name, value):
    def initializer(ignored_rng_key):
        return value
    return module.param(name, init_fn=initializer)


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

        self.time_first = simple_param(self, 'time_first', time_first)

        x = (jnp.arange(embedding_size) /
             embedding_size)[jnp.newaxis, :]
        self.time_mix_k = simple_param(self,'time_mix_k', jnp.power(x, ratio_1_to_almost_0))
        self.time_mix_v = simple_param(self,'time_mix_v', self.time_mix_k + .3 * ratio_0_to_1)
        self.time_mix_r = simple_param(self,'time_mix_r', jnp.power(x, .5 * ratio_1_to_almost_0))

        h = jnp.arange(0, embedding_size)
        decay_speed = -5 + 8 * (h / (embedding_size - 1)
                                ) ** (.7 + 1.3 * ratio_0_to_1)
        self.time_decay = simple_param(self, 'time_decay', decay_speed)
        self.key = nn.Dense(embedding_size, use_bias=False)
        self.value = nn.Dense(embedding_size, use_bias=False)
        self.receptance = nn.Dense(embedding_size, use_bias=False)
        self.output = nn.Dense(embedding_size, use_bias=False)


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
        rwkv = sr * wkv(self.time_decay, self.time_first, k, v).T
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
        self.key = nn.Dense(hidden_size, use_bias=False)
        self.receptance = nn.Dense(embedding_size, use_bias=False)
        self.value = nn.Dense(embedding_size, use_bias=False)

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
    mix1_type: Union[Type[ChannelMix], Type[TimeMix]]  = TimeMix

    def setup(self):
        config, layer_num = self.config, self.layer_num
        embedding_size = config.embedding_size
        position_embedding_size = config.pos_embedding_size
        self.layer_norm1 = nn.LayerNorm(epsilon=1e-5)
        self.layer_norm2 = nn.LayerNorm(epsilon=1e-5)
        self.mix1 = self.mix1_type(config, layer_num)
        self.mix2 = ChannelMix(config, layer_num)

    @nn.jit
    def __call__(self, x, key=None):
        T, C = x.shape
        x = x + self.mix1(self.layer_norm1(x))
        x = x + self.mix2(self.layer_norm2(x))
        return x

class TinyAttention(nn.Module):
    config:Config

    def setup(self):
        embedding_size, attention_size = self.config.embedding_size, self.config.attention_size
        self.layernorm = nn.LayerNorm(epsilon=1e-5)
        self.mask = jnp.tril(jnp.ones((self.config.context_length, self.config.context_length))) == 1
        self.q = nn.Dense(attention_size, use_bias=False)
        self.k = nn.Dense(attention_size, use_bias=False)
        self.v = nn.Dense(embedding_size, use_bias=False)

    @nn.jit
    def __call__(self, x, x_emb):
        T, C = x.shape
        xx = self.layernorm(x)
        q = self.q(xx)[:T, :]
        k = self.k(xx)[:T, :]
        c = (q @ k.T) * (self.config.attention_size ** (-0.5))
        c = jnp.where(self.mask[:T,:T], c, 0)
        x = x + (c @ self.v(x_emb))
        return x

class PositionEmbedding(nn.Module):
    config:Config

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
        self.head_q = nn.Dense(self.config.head_qk_size, use_bias=False)
        self.head_k = nn.Dense(self.config.head_qk_size, use_bias=False)
        context_length = self.config.context_length
        self.copy_mask = jnp.tril(jnp.ones((context_length, context_length))) == 1
    
    @nn.jit
    def __call__(self, x, idx):
        T, C = x.shape
        q = self.head_q(x)[:T, :]
        k = self.head_k(x)[:T, :]
        c = (q @ k.T) * (1.0 / (self.config.head_qk_size))
        c = jnp.where(self.copy_mask[:T, :T], c, 0)
        one_hot = nn.one_hot(idx, num_classes=self.config.vocab_size).astype(x.dtype)
        return c @ one_hot


class RWKV(nn.Module):
    config: Config

    def setup(self):
        config = self.config
        self.emb = nn.Embed(config.vocab_size, config.embedding_size)
        self.ln_in = nn.LayerNorm(epsilon=1e-5)

        if  0 <= config.attention_at_layer < config.num_layers:
            self.tiny_attention = TinyAttention(config)

        blocks = []
        for i in range(config.num_layers):
            mix = ChannelMix if i == 0 and config.channel_mix_at_input else TimeMix
            blocks.append(Block(config, i, mix))
        self.blocks = blocks

        self.ln_out = nn.LayerNorm(epsilon=1e-5)
        self.head = nn.Dense(config.vocab_size, use_bias=False)


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