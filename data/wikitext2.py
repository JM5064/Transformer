import collections
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import bpe

class WikiText2(Dataset):

    def __init__(self, split='train'):
        super().__init__()

        if split == 'train':
            self.df = pd.read_parquet('data/wikitext2/train-00000-of-00001.parquet')
        elif split == 'val':
            self.df = pd.read_parquet('data/wikitext2/validation-00000-of-00001.parquet')
        else:
            self.df = pd.read_parquet('data/wikitext2/test-00000-of-00001.parquet')

        self.create_data(min_length=50, num_merges=100)
        
    def create_data(self, min_length, num_merges):
        # convert to array of arrays
        self.arr = self.df.to_numpy()

        # merge all text values
        all_text = ""
        for item in self.arr:
            text = item[0]
            if text == "" or len(text) < min_length:
                continue
            stripped_text = text.strip("\n")
            all_text += stripped_text
            all_text = all_text.strip()

        self.tokenized_text, self.vocab = bpe.bpe(all_text, num_merges)
        print(self.tokenized_text)
        print(self.vocab)


if __name__ == "__main__":
    dataset = WikiText2()



