import pandas as pd
from torch.utils.data import Dataset


class WikiText2(Dataset):

    def __init__(self, split='train'):
        super().__init__()

        if split == 'train':
            self.df = pd.read_parquet('data/wikitext2/train-00000-of-00001.parquet')
        elif split == 'val':
            self.df = pd.read_parquet('data/wikitext2/validation-00000-of-00001.parquet')
        else:
            self.df = pd.read_parquet('data/wikitext2/test-00000-of-00001.parquet')

        self.data = []
        self.create_data(min_length=50)
        
        for i in range(len(self.data)):
            print(self.data[i])
            print()


    def create_data(self, min_length):

        for row in self.df.itertuples():
            text = row.text

            if len(text) < min_length:
                continue

            stripped_text = text.strip('\n')

            self.data.append(stripped_text)


    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    dataset = WikiText2()



