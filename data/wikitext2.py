import collections
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset

import bpe

class WikiText2(Dataset):

    def __init__(self, split='train', seq_len=128, 
                 vocab_csv='data/wikitext2/vocab.json', encoded_text_csv='data/wikitext2/encoded_text.json'         
        ):
        super().__init__()

        if split == 'train':
            self.df = pd.read_parquet('data/wikitext2/train-00000-of-00001.parquet')
        elif split == 'val':
            self.df = pd.read_parquet('data/wikitext2/validation-00000-of-00001.parquet')
        else:
            self.df = pd.read_parquet('data/wikitext2/test-00000-of-00001.parquet')

        self.seq_len = seq_len
        self.vocab_csv = vocab_csv
        self.encoded_text_csv = encoded_text_csv

        self.create_data(min_length=50, num_merges=5)


    def __getitem__(self, index):
        # if files dont exist, create them
        # otherwise, load them
        # encode text

        input_tokens = self.encoded_text[index : index + self.seq_len]
        target_tokens = self.encoded_text[index + 1 : index + self.seq_len + 1]

        return input_tokens, target_tokens
    

    def __len__(self):
        return len(self.encoded_text) - self.seq_len
    
        
    def create_data(self, min_length, num_merges):
        # convert to array of paragraphs
        text = self.df.loc[:,"text"].to_list()
        
        # remove empty/very short paragraphs
        self.array_text = [t.strip("\n") for t in text if len(t) > min_length and t != ""]

        # Run bpe
        tokenized_text, self.vocab = bpe.bpe(self.array_text, num_merges)

        # Create encoding and decoding maps
        self.encoding, self.decoding = bpe.make_mapping(self.vocab)

        # Encode text
        self.encoded_text = self.encode_text(tokenized_text)
        
        # Save vocab and encoded text
        self.save_vocab(self.vocab, self.vocab_csv)
        self.save_encoded_text(self.encoded_text, self.encoded_text_csv)

    
    def save_vocab(self, vocab, vocab_file):
        with open(vocab_file, "w", encoding="utf-8") as file:
            json.dump(vocab, file, ensure_ascii=False, indent=2)


    def load_vocab(self, vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as file:
            vocab = json.load(file)

        return vocab
    

    def encode_text(self, tokenized_text):
        encoded_text = []
        for token in tokenized_text:
            encoded_text.append(self.encoding[token])

        return encoded_text
    

    def save_encoded_text(self, encoded_text, encoded_text_file):
        with open(encoded_text_file, "w", encoding="utf-8") as file:
            json.dump(encoded_text, file, ensure_ascii=False)


    def load_encoded_text(self, encoded_text_file):
        with open(encoded_text_file, "r", encoding="utf-8") as file:
            encoded_text = json.load(file)

        return encoded_text


if __name__ == "__main__":
    dataset = WikiText2()