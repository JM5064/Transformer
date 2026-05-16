import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, seq_len=128, d_model=512, n=10000):
        super().__init__()

        positional_encoding = torch.zeros((seq_len, d_model))

        for pos in range(seq_len):
            for i in range(d_model // 2):
                denominator = torch.tensor(n ** (2*i/d_model))
                positional_encoding[pos, 2*i] = torch.sin(pos/denominator)
                positional_encoding[pos, 2*i+1] = torch.cos(pos/denominator)

        self.positional_encoding = nn.Buffer(positional_encoding)


    def forward(self, x):
        x = x + self.positional_encoding

        return x


if __name__ == "__main__":
    PositionalEncoding(seq_len=4, d_model=4, n=100)
