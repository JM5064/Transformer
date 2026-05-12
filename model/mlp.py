import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, d_model=512, seq_len=128):
        super().__init__()

        self.layer1 = nn.Linear(in_features=d_model, out_features=2048)
        self.layer2 = nn.Linear(in_features=2048, out_features=d_model)

        self.layer_norm = nn.LayerNorm((d_model, seq_len))


    def forward(self, X):
        X_resid = X

        X = torch.transpose(X, 0, 1)

        X = self.layer1(X)
        X = F.relu(X)
        X = self.layer2(X)

        X = torch.transpose(X, 0, 1)

        X = X + X_resid

        X = self.layer_norm(X)

        return X