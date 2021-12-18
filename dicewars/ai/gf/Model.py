# python3
import numpy as np
import torch as t
import torch.nn.functional as F


class Model(t.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = t.nn.Linear(10, 1)

    def forward(self, x):
        return np.squeeze(t.sigmoid(self.linear(x)), axis=1)

    def prob_class_1(self, x):
        prob = self(t.from_numpy(x.astype(np.float32)))
        return prob.detach().numpy()
