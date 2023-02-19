from dataclasses import dataclass

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