import math
from dataclasses import dataclass, field
from functools import partial
from itertools import repeat
from typing import Any, List, Optional

import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass
class Config:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    embedding_size: int = field(
        default=1024,
        metadata={"help": "The size of the embedding dimension"},
    )
    num_layers: int = field(
        default=10,
        metadata={"help": "The number of block layers in the model."},
    )
    vocab_size: int = field(
        default=0,
        metadata={"help": "(Only for vision experiments): size of vocabulary"},
    )
    context_length: int = field(
        default=1024,
        metadata={"help": "The number of tokens in the context"},
    )
    attention_at_layer: int = field(
        default=-1,
        metadata={
            "help": "which layer we should add a small attention too. If negative, no attention is included."},
    )
    attention_size: int = field(
        default=-1,
        metadata={"help": "Size of the optional attention."},
    )
    head_qk_size: int = field(
        default=0,
        metadata={"help": "Size of the head QK"},
    )
    channel_mix_at_input: bool = field(
        default=False,
        metadata={
            "help": "whether we should use a channel mix instead of time mix at the input"},
    )
    pos_embedding_size: int = field(
        default=0,
        metadata={
            "help": "(Only for vision experiments): size of position embedding"},
    )
    dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )


def initialize_to_value(x, dtype):
    """
    makes an initializer function that ignores the given PRNGKey
    and always returns the given value
    """
    return lambda _: x.astype(dtype)


class TimeMix(nn.Module):
    layer_depth: int
    config: Config

    def setup(self):
        layer_depth = self.layer_depth
        embedding_size = self.config.embedding_size
        num_layers = self.config.num_layers
        dtype = self.config.dtype

        # goes from 0 to 1 along layer depth
        ratio_0_to_1 = layer_depth / (num_layers - 1)
        # goes from 1 to (almost) 0 along layer depth
        ratio_1_to_almost_0 = 1.0 - (layer_depth / num_layers)
        zigzag = .5 * (jnp.arange(1, embedding_size+1) % 3 - 1)
        time_first = jnp.full(embedding_size, math.log(.3)) + zigzag
        self.time_first = self.param(
            'time_first', initialize_to_value(time_first, 'float32'))

        h = jnp.arange(0, embedding_size)
        # the numbers used here were found to work well from experiments
        time_decay = -5 + 8 * (h / (embedding_size - 1)
                               ) ** (.7 + 1.3 * ratio_0_to_1)

        self.time_decay = self.param(
            'time_decay', initialize_to_value(time_decay, 'float32'))

        x = (jnp.arange(embedding_size) /
             embedding_size)
        time_mix_k = jnp.power(x, ratio_1_to_almost_0)
        time_mix_v = time_mix_k + .3 * ratio_0_to_1
        time_mix_r = jnp.power(x, .5 * ratio_1_to_almost_0)
        self.time_mix_k = self.param(
            'time_mix_k', initialize_to_value(time_mix_k, dtype))
        self.time_mix_v = self.param(
            'time_mix_v', initialize_to_value(time_mix_v, dtype))
        self.time_mix_r = self.param(
            'time_mix_r', initialize_to_value(time_mix_r, dtype))

        self.layernorm = nn.LayerNorm(epsilon=1e-5, dtype=dtype)
        self.key = nn.Dense(embedding_size, use_bias=False, dtype=dtype)
        self.value = nn.Dense(embedding_size, use_bias=False, dtype=dtype)
        self.receptance = nn.Dense(embedding_size, use_bias=False, dtype=dtype)
        self.output = nn.Dense(embedding_size, use_bias=False, dtype=dtype)

    def __call__(self, x, time_mix_state):
        sx, aa, bb, pp = time_mix_state
        xx = self.layernorm(x)
        sx = jnp.concatenate((jnp.expand_dims(sx, 0), xx[:-1, :]))
        kx = xx * self.time_mix_k + sx * (1 - self.time_mix_k)
        vx = xx * self.time_mix_v + sx * (1 - self.time_mix_v)
        rx = xx * self.time_mix_r + sx * (1 - self.time_mix_r)
        r = nn.sigmoid(self.receptance(rx))
        k = self.key(kx)
        v = self.value(vx)
        T = x.shape[0]

        def step(state, kv):
            (aa, bb, pp), (kk, vv) = state, kv
            ww = self.time_first + kk
            p = jnp.maximum(pp, ww)
            e1 = jnp.exp(pp - p)
            e2 = jnp.exp(ww - p)
            out = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).astype(dtype=r.dtype)

            ww = -jnp.exp(self.time_decay) + pp
            p = jnp.maximum(ww, kk)
            e1 = jnp.exp(ww - p)
            e2 = jnp.exp(kk - p)
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p
            return (aa, bb, pp), out

        (aa, bb, pp), sx = jax.lax.scan(step, (aa, bb, pp), (k, v))
        out = x + self.output(r * sx)
        # (xx[-1,:], aa, bb, pp) is the next time_mix_state
        return out, (xx[-1, :], aa, bb, pp)


class ChannelMix(nn.Module):
    layer_depth: int
    config: Config

    def setup(self):
        layer_depth = self.layer_depth
        num_layers = self.config.num_layers
        embedding_size = self.config.embedding_size
        dtype = self.config.dtype

        hidden_size = 4 * embedding_size
        self.layernorm = nn.LayerNorm(epsilon=1e-5, dtype=dtype)
        self.key = nn.Dense(hidden_size, use_bias=False, dtype=dtype)
        self.receptance = nn.Dense(embedding_size, use_bias=False, dtype=dtype)
        self.value = nn.Dense(embedding_size, use_bias=False, dtype=dtype)

        x = (jnp.arange(embedding_size) /
             embedding_size)

        ratio_1_to_almost_0 = 1.0 - (layer_depth / num_layers)
        time_mix_k = jnp.power(x, ratio_1_to_almost_0)
        time_mix_r = jnp.power(x, .5 * ratio_1_to_almost_0)
        self.time_mix_k = self.param(
            'time_mix_k', initialize_to_value(time_mix_k, dtype))
        self.time_mix_r = self.param(
            'time_mix_r', initialize_to_value(time_mix_r, dtype))

    def __call__(self, x, channel_mix_state):
        xx = self.layernorm(x)
        sx = jnp.concatenate(
            (jnp.expand_dims(channel_mix_state, 0), xx[:-1, :]))
        xk = xx * self.time_mix_k + sx * (1 - self.time_mix_k)
        xr = xx * self.time_mix_r + sx * (1 - self.time_mix_r)
        r = nn.sigmoid(self.receptance(xr))
        k = jnp.square(nn.relu(self.key(xk)))
        out = x + r * self.value(k)

        return out, xx[-1, :]

def empty_state(embedding_size):
    "returns an empty block_state for a given embedding size"
    zeros = jnp.zeros(embedding_size)
    min_values = jnp.full(embedding_size, -jnp.inf)
    time_mix_state = (zeros, zeros, zeros, min_values)
    channel_mix_state = zeros
    return time_mix_state, channel_mix_state

class Block(nn.Module):
    layer_num: int
    config: Config

    def setup(self):
        self.time_mix = TimeMix(self.layer_num, self.config)
        self.channel_mix = ChannelMix(self.layer_num, self.config)

    def __call__(self, x, block_state):
        """
        Takes the embedding from the previous layer, and the `block_state`
        from the previous time_step.

        `block_state` is a tuple of `time_mix_state` and `channel_mix_state`,
        which are used as inputs to the block's `time_mix` and `channel_mix`
        respectively.
        """
        if block_state is None:
            block_state = empty_state(self.config.embedding_size)

        time_mix_state, channel_mix_state = block_state
        x, time_mix_state = self.time_mix(x, time_mix_state)
        x, channel_mix_state = self.channel_mix(x, channel_mix_state)
        return x, (time_mix_state, channel_mix_state)


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
        self.head_q = nn.Dense(self.config.head_qk_size,
                               use_bias=False, dtype=config.dtype)
        self.head_k = nn.Dense(self.config.head_qk_size,
                               use_bias=False, dtype=config.dtype)
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
        self.embed = nn.Embed(config.vocab_size,
                              config.embedding_size,
                              config.dtype)
        self.input_layernorm = nn.LayerNorm(epsilon=1e-5, dtype=config.dtype)
        self.blocks = [Block(i, self.config) for i in range(config.num_layers)]
        self.output_layernorm = nn.LayerNorm(epsilon=1e-5, dtype=config.dtype)
        self.head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(self, x, block_states=None, output_state=False):
        # we use a helper method for the forward pass until
        # nn.jit supports static_argnames
        return self.forward_with_state(x, block_states, output_state)
    
    @partial(nn.jit, static_argnums=3)
    def forward_with_state(self, x, block_states, output_state):
        x = self.embed(x)
        x = self.input_layernorm(x)

        next_states = []
        if block_states is None:
            block_states = repeat(None)
        for block, state in zip(self.blocks, block_states):
            x, new_state = block(x, state)
            if output_state:
                next_states.append(new_state)

        x = self.output_layernorm(x)
        x = self.head(x)

        if output_state:
            return x, next_states
        return x

BatchRWKV = nn.vmap(RWKV, in_axes=0, out_axes=0,
                    variable_axes={ 'params' : None },
                    split_rngs={ 'params' : False })

def save_weights(output_path, params, step):
    checkpoints.save_checkpoint(output_path, params, step)


def load_weights(path:str, dtype='float32', embedding_padding=0):
    import torch
    ws = {k: v.float() for k, v in torch.load(path, 'cpu').items()}
    out = {}
    for path, weight in ws.items():
        path = path.replace('emb.weight', 'embed.embedding')
        if 'ln' in path:
            path = path.replace('weight', 'scale')
        else:
            path = path.replace('weight', 'kernel')

        path = path.replace('blocks.0.ln0', 'input_layernorm')
        path = path.replace('ln1', 'time_mix.layernorm')
        path = path.replace('ln2', 'channel_mix.layernorm')
        path = path.replace('ffn', 'channel_mix')
        path = path.replace('att', 'time_mix')
        path = path.replace('ln_out', 'output_layernorm')
        path = path.replace('blocks.', 'blocks_')
        if 'time_mix_' in path and len(weight.shape) > 1:
            weight = weight.resize(weight.shape[-1])
        if 'kernel' in path and len(weight.shape) > 1:
            weight = weight.T

        if 'time_decay' in path or 'time_first' in path:
            dtype = 'float32'
        if 'embedding' in path:
            V, C = weight.shape
            padded = torch.zeros(V+embedding_padding, C)
            padded[:V] = weight
            weight = padded
        if 'head' in path:
            C, V = weight.shape
            padded = torch.zeros(C, V+embedding_padding)
            padded[:, :V] = weight
            weight = padded
        out[path] = jnp.array(weight.detach().numpy(), dtype)

    out = {'params': flax.traverse_util.unflatten_dict(out, sep='.')}
    return out