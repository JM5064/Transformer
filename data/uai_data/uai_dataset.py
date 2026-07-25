import torch
import json
import os
from torch.utils.data import Dataset


class UAIDataset(Dataset):


    def __init__(self, split_path):
        super().__init__()

        self.members = ['carman', 'dzss', 'friendlynoob', 'genzi', 'heibunny', 'jmoreojm', 'msn', 'rmj', 'tcray', 'willywonka']

        self.members_data = [self.load_json(os.path.join(split_path, f'{member}.json')) for member in self.members]

        # Put all members_data in a big list
        data = []
        for member_data in self.members_data:
            data.extend(member_data)
        
        # Put all labels in a big list
        labels = []
        for i, member_data in enumerate(self.members_data):
            labels.extend([i] * len(member_data))

        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)
        

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    

    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data
    

if __name__ == "__main__":
    uai_dataset = UAIDataset('data/uai_data/uai_dataset/train')

    print(uai_dataset[2387])

    
