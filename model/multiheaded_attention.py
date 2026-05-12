import torch
import torch.nn as nn


class MultiheadedAttention(nn.Module):

    def __init__(self, d_k=64, d_v=64, d_model=512, num_heads=8, mask=False):
        super().__init__()

        self.d_k = d_k
        self.mask = mask

        self.W_Q = nn.Parameter(torch.zeros((d_k * num_heads), d_model))
        self.W_K = nn.Parameter(torch.zeros((d_k * num_heads), d_model))

        self.W_output = nn.Parameter(torch.zeros(d_model, d_v * num_heads)) # value up
        self.W_V = nn.Parameter(torch.zeros(d_v * num_heads, d_model))      # value down
        # (d_k x num_heads x d_model) x 4 params

        # TODO: layer norm

        self.initialize_weights()


    def initialize_weights(self):
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_output)
        nn.init.xavier_uniform_(self.W_V)

    
    def forward(self, X):
        # Input: embedding length (d_model) * seq_len
        X_resid = X

        # Compute query, key, and value matrices
        Q = self.W_Q @ X
        K = self.W_K @ X
        V = self.W_V @ X

        # Compute attention
        before_softmax = (Q @ K.T) / torch.sqrt(self.d_k)
        after_softmax = torch.softmax(before_softmax, dim=0)

        attention = self.W_output @ (after_softmax @ V)

        X = attention + X_resid

        return X
    

if __name__ == "__main__":
    mha = MultiheadedAttention()

    total_params = sum(p.numel() for p in mha.parameters())
    print(total_params)
