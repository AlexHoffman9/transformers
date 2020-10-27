#neuron_mask.py
'''
Layer to mask the activations after any type of layer.
'''
class NeuronMask(nn.module):
    def __init__(self):
        super(NeuronMask, self)._init__(n_features)
        self.mask = torch.ones()