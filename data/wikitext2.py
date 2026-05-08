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
        # convert to array of paragraphs
        temp_arr = self.df.loc[:,"text"].to_list()
        
        # remove empty/very short paragraphs
        self.arr = [t.strip("\n") for t in temp_arr if len(t) < min_length and t != ""]

        self.tokenized_text, self.vocab = bpe.bpe(self.arr, num_merges)
        
        print(self.tokenized_text)
        print(self.vocab)


if __name__ == "__main__":
    dataset = WikiText2()



