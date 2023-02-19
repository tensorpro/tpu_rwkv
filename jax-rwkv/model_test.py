import unittest
import model
import torch
import numpy as np
import convert

device = 'cpu'

def check_keys_match(a, b, match_fn):
    for k, v in b.items():
        if v == None:
            continue
        else:
            match_fn(v, b[k])


# testdata has serialized dictionaries, each with the
# following schema:
# 
# `model_weights` is a state_dict for a torch module
# `input` is an examlpe input to the torch module
# `output` is the model's output when given input

class TestModel(unittest.TestCase):

    def test_model(self):
        # loads test data with a torch model with:
        #    tiny_att_dim=7, tiny_att_layer=5, head_qk=11
        #    vocab_size=6, n_layer=10, n_embd=5, pre_ffn=1
        test_data = torch.load('./testdata/small-model-test.pt')
        params, config = convert.rwkv(test_data['model_state_dict'])
        rwkv = model.BatchRWKV(config)
        jax_out = rwkv.apply(params, test_data['input'].numpy())
        np.testing.assert_allclose(
            test_data['output'].detach().numpy(), np.array(jax_out), rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
