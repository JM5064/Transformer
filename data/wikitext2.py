import pandas as pd
from torch.utils.data import Dataset


class WikiText2(Dataset):

    def __init__(self, split='train'):
        super().__init__()

        if split == 'train':
            self.df = pd.read_parquet('data/wikitext2/train-00000-of-00001.parquet').to_dict()
        elif split == 'val':
            self.df = pd.read_parquet('data/wikitext2/validation-00000-of-00001.parquet').to_dict()
        else:
            self.df = pd.read_parquet('data/wikitext2/test-00000-of-00001.parquet').to_dict()


    def clean(self):
        for i, text in enumerate(self.df.iterrows()):
            print(text)
            
            if i > 5:
                break


if __name__ == "__main__":
    dataset = WikiText2()

    dataset.clean()

