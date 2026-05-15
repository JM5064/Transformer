import torch
import torch.nn as nn


class MultiheadedAttention(nn.Module):

    def __init__(self, d_k=64, d_v=64, d_model=512, num_heads=8, seq_len=128, use_mask=False):
        super().__init__()

        self.d_k = d_k
        self.use_mask = use_mask
        
        self.W_Q = nn.Parameter(torch.zeros(d_model, d_k * num_heads))
        self.W_K = nn.Parameter(torch.zeros(d_model, d_k * num_heads))

        self.W_output = nn.Parameter(torch.zeros(d_v * num_heads, d_model)) # value up
        self.W_V = nn.Parameter(torch.zeros(d_model, d_v * num_heads))      # value down
        # (d_k x num_heads x d_model) x 4 params

        self.layer_norm = nn.LayerNorm((seq_len, d_model))
        # (d_model x seq_len) x 2 params

        if self.use_mask:
            self.mask = torch.zeros((seq_len, seq_len))
            for i in range(seq_len):
                for j in range(i):
                    self.mask[i, j] = -torch.inf

        self.initialize_weights()


    def initialize_weights(self):
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_output)
        nn.init.xavier_uniform_(self.W_V)

    
    def forward(self, X, Y):
        """X : input of size (batch, seq_len, d_model)"""

        X_resid = Y

        # Compute query, key, and value matrices
        Q = Y @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # Tranpose keys (not including batch dim)
        K_T = torch.transpose(K, -2, -1)

        # Compute attention
        before_softmax = (Q @ K_T) / (self.d_k ** 0.5)

        if self.use_mask:
            before_softmax = before_softmax + self.mask
        
        after_softmax = torch.softmax(before_softmax, dim=-1)

        attention = (after_softmax @ V) @ self.W_output

        # Add residual connection, and layer norm
        X = attention + X_resid

        X = self.layer_norm(X)

        return X
    

if __name__ == "__main__":
    input_data1 = torch.randn((1, 4, 4))
    input_data2 = torch.randn((1, 4, 4))

    mha = MultiheadedAttention(d_k=1, d_v=1, d_model=4, seq_len=4, use_mask=True)

    mha(input_data1, input_data2)
    