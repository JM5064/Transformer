import torch.nn as nn
from model.positional_encoding import PositionalEncoding
from model.encoder_block import EncoderBlock
from model.decoder_block import DecoderBlock
import time


class OptimusPrime(nn.Module):

    def __init__(self, vocab_size, d_model=512):
        super().__init__()

        self.positional_encoding = PositionalEncoding()

        self.input_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.output_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.encoder = nn.ModuleList()
        for _ in range(6):
            self.encoder.append(EncoderBlock())

        self.decoder = nn.ModuleList()
        for _ in range(6): 
            self.decoder.append(DecoderBlock())

        self.unembedding = nn.Linear(in_features=d_model, out_features=vocab_size)


    def forward(self, X, Y):
        X = self.input_embedding(X)
        X = self.positional_encoding(X)

        Y = self.output_embedding(Y)
        Y = self.positional_encoding(Y)

        for layer in self.encoder:
            X = layer(X)

        for layer in self.decoder:
            Z = layer(X, Y)

        Z = self.unembedding(Z)

        return Z


if __name__ == "__main__":
    from data.wikitext2 import WikiText2
    from torch.utils.data.dataloader import DataLoader


    wikitext2 = WikiText2()

    dl = DataLoader(wikitext2, batch_size=4)
    op = OptimusPrime(vocab_size=10217)

    for X, Y in dl:
        s = time.time()
        op(X, Y)
        print((time.time() - s) * 1000)

        break




