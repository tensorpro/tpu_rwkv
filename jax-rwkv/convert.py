"""
each function here is used to convert a torch rwkv component
to a jax equivalent.

this really should really convert serialized state_dicts directly,
but it was convenient to access fields in the torch modules in some
of these functions

I'm using this to make test cases validating the equivalence of torch
vs jax implementations.
"""
import jax.numpy as jnp
from config import Config

def param_to_np_array(torch_param):
    return torch_param.detach().numpy()

def join_path(prefix, suffix):
    if prefix != '':
        prefix += '.'
    return f'{prefix}{suffix}'

def param_to_array(torch_param):
    return jnp.array(param_to_np_array(torch_param))

def layernorm(state_dict, prefix=''):
    vars = {}
    vars['scale'] = param_to_array(state_dict[join_path(prefix, 'weight')])
    vars['bias'] = param_to_array(state_dict[join_path(prefix, 'bias')])
    return vars


def linear(state_dict, prefix=''):
    vars = {}
    vars['kernel'] = param_to_array(state_dict[join_path(prefix, 'weight')].T)
    return vars

def time_mix(state_dict, prefix=''):
    vars = {}
    for copied in ['time_mix_k', 'time_mix_v', 'time_mix_r', 'time_first', 'time_decay']:
        val = state_dict[join_path(prefix, copied)]
        if 'time_mix' in copied:
            val = val[0] # clears out the empty batch dimension
        vars[copied] = param_to_array(val)
    for linear_transform in ['key', 'value', 'receptance', 'output']:
        vars[linear_transform] = linear(state_dict, join_path(prefix,linear_transform))
    return vars

def channel_mix(state_dict, prefix=''):
    vars = {} 
    for copied in ['time_mix_k', 'time_mix_r']:
        vars[copied] = param_to_array(state_dict[join_path(prefix, copied)])
    for linear_transform in ['key', 'value', 'receptance']:
        vars[linear_transform] = linear(state_dict, join_path(prefix, linear_transform))
    return vars


def block(state_dict, prefix=''):
    vars = {}
    for i in range(1, 3):
        vars[f'layer_norm{i}'] = layernorm(state_dict, join_path(prefix,f'ln{i}'))
    try:
        vars['mix1'] = time_mix(state_dict, join_path(prefix,'att'))
    except:
        vars['mix1'] = channel_mix(state_dict, join_path(prefix,'ffnPre'))
    vars['mix2'] = channel_mix(state_dict, join_path(prefix,'ffn'))
    return vars


def tiny_attention(state_dict, prefix=''):
    vars = {}
    vars['layernorm'] = layernorm(state_dict, join_path(prefix, 'tiny_ln'))
    for linear_transform in ['v', 'q', 'k']:
        vars[linear_transform] = linear(state_dict, join_path(prefix, f'tiny_{linear_transform}'))
    return vars


def infer_config(state_dict):
    attention_at_layer = -1
    tiny_att_size = -1
    num_layers = int(max(k for k in state_dict.keys() if 'blocks' in k).split('.')[1])
    for i in range(num_layers):
        if f'blocks{i}.tiny_q.weight' in state_dict.keys():
            attention_at_layer=i
            tiny_att_size = state_dict['blocks{i}.tiny_q.weight'].shape[0]


    return Config(num_layers=num_layers, attention_size=tiny_att_size, attention_at_layer=attention_at_layer)

def head_qk(state_dict, prefix=''):
    vars = {}
    vars['head_q'] = linear(state_dict,join_path(prefix, 'head_q'))
    vars['head_k'] = linear(state_dict,join_path(prefix, 'head_k'))
    return vars

def embedding(state_dict, prefix=''):
    vars = {}
    vars['embedding'] = param_to_array(state_dict[join_path(prefix, 'weight')])
    return vars

def rwkv(state_dict):
    prefix = ''
    vars = {}
    channel_mix_at_input = 'blocks.0.ffnPre.time_mix_k' in state_dict.keys()
    num_layers = max(int(k.split('.')[1]) for k in state_dict.keys() if 'blocks.' in k) + 1
    vocab_size, embedding_size = state_dict['emb.weight'].shape
    attention_at_layer = -1
    attention_size = -1
    head_qk_size = 0
    vars['emb'] = embedding(state_dict, 'emb')
    vars['ln_in'] = layernorm(state_dict, 'blocks.0.ln0')
    vars['ln_out'] = layernorm(state_dict, 'ln_out')
    vars['head'] = linear(state_dict, 'head')
    for i in range(num_layers):
        vars[f'blocks_{i}'] = block(state_dict, f'blocks.{i}')
        if f'blocks.{i}.tiny_q.weight' in state_dict.keys():
            attention_at_layer=i
            attention_size = state_dict[f'blocks.{i}.tiny_q.weight'].shape[0]
            vars['tiny_attention'] = tiny_attention(state_dict, f'blocks.{i}')
    if 'head_q.weight' in state_dict.keys():
        vars['head_qk'] = head_qk(state_dict, '')
        head_qk_size = state_dict['head_q.weight'].shape[0]
    return { 'params': vars }, Config(
        pos_embedding_size=0,
        channel_mix_at_input=channel_mix_at_input,
        num_layers=num_layers,
        embedding_size=embedding_size,
        vocab_size=vocab_size,
        attention_at_layer=attention_at_layer,
        attention_size=attention_size,
        head_qk_size=head_qk_size,
    )