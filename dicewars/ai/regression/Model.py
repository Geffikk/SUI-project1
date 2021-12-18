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


if __name__ == '__main__':
    linear = t.nn.Linear(8, 1)
    x = t.Tensor([[1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 3, 4, 5, 6, 7, 8]])
    print(x)
    print(t.sigmoid(linear(x)))
    # np.squeeze vytvori 1-D pole z vysledkami cize [[x,y,z,...]] => [x,y,z,...]
    # tensor([[0.7174], [0.6544]], grad_fn=<SigmoidBackward0>) => tensor([0.7174, 0.6544], grad_fn=<SqueezeBackward1>)
    # sigmoid vrati hodnotu medzi 0 a 1
    print(np.squeeze(t.sigmoid(linear(x)), axis=1))
