import unittest
import model
import torch
import numpy as np
from jax.random import PRNGKey
import convert

device = 'cpu'
from importlib import reload
reload(model)

# the jax-rwkv/testdata directoryhas serialized dictionaries, each with the
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
        test_data = torch.load(f'./jax-rwkv/testdata/small-model-test.pt')
        params, config = convert.rwkv(test_data['model_state_dict'])
        rwkv = model.BatchRWKV(config)
        x = test_data['input'].numpy() # x = [[0,1,2]]
        params['cache'] = rwkv.init(PRNGKey(0), x)['cache']
        jax_out = rwkv.apply(params, x)

        np.testing.assert_allclose(
            test_data['output'].detach().numpy(), np.array(jax_out), rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
