import pandas as pd
import json
import os
from special_chars import UNK
from torch.utils.data import Dataset

import bpe

class WikiText2(Dataset):

    def __init__(self, split='train', seq_len=128, 
            encoded_text_json='data/wikitext2/encoded_text.json',
            vocab_json='data/wikitext2/vocab.json', 
            merge_pairs_json='data/wikitext2/merge_pairs.json'
        ):
        super().__init__()

        if split == 'train':
            self.df = pd.read_parquet('data/wikitext2/train-00000-of-00001.parquet')
        elif split == 'val':
            self.df = pd.read_parquet('data/wikitext2/validation-00000-of-00001.parquet')
        else:
            self.df = pd.read_parquet('data/wikitext2/test-00000-of-00001.parquet')

        self.split = split
        self.seq_len = seq_len

        self.vocab_json = vocab_json
        self.encoded_text_json = encoded_text_json
        self.merge_pairs_json = merge_pairs_json

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
        # If encoded text file exists, load it. Otherwise, run BPE
        if self.file_exists(self.encoded_text_json):
            return self.load_encoded_text(self.encoded_text_json)

        # convert to array of paragraphs
        text = self.df.loc[:,"text"].to_list()
        
        # remove empty/very short paragraphs
        self.array_text = [t.strip("\n") for t in text if len(t) > min_length and t != ""]

        if self.split == 'train':
            # Run bpe
            tokenized_text, self.vocab, merge_pairs = bpe.bpe(self.array_text, num_merges)

            # Create encoding and decoding maps
            self.encoding, self.decoding = bpe.make_mapping(self.vocab)

            # Encode text
            self.encoded_text = self.encode_text(tokenized_text)
            
            # Save vocab, encoded text, and merge pairs
            self.save_vocab(self.vocab, self.vocab_json)
            self.save_encoded_text(self.encoded_text, self.encoded_text_json)
            self.save_merge_pairs(merge_pairs, self.merge_pairs_json)
        else:
            # Load merge pair file and vocabulary
            # NOTE: these files need to be created by running the split as 'train' first
            merge_pairs = self.load_merge_pairs(self.merge_pairs_json)
            self.vocab = self.load_vocab(self.vocab_json)

            # Run merges
            tokenized_text = bpe.apply_merge_pairs(self.array_text, merge_pairs)

            # Create encoding and decoding maps
            self.encoding, self.decoding = bpe.make_mapping(self.vocab)

            # Encode text
            self.encoded_text = self.encode_text(tokenized_text)
            self.save_encoded_text(self.encoded_text, self.encoded_text_json)

        return self.encoded_text

    
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
            if token in self.encoding:
                encoded_text.append(self.encoding[token])
            else:
                encoded_text.append(UNK)

        return encoded_text
    

    def save_encoded_text(self, encoded_text, encoded_text_file):
        with open(encoded_text_file, "w", encoding="utf-8") as file:
            json.dump(encoded_text, file, ensure_ascii=False)


    def load_encoded_text(self, encoded_text_file):
        with open(encoded_text_file, "r", encoding="utf-8") as file:
            encoded_text = json.load(file)

        return encoded_text
    

    def save_merge_pairs(self, merge_pairs, merge_pairs_file):
        with open(merge_pairs_file, "w", encoding="utf-8") as file:
            json.dump(merge_pairs, file, ensure_ascii=False)


    def load_merge_pairs(self, merge_pairs_file):
        with open(merge_pairs_file, "r", encoding="utf-8") as file:
            merge_pairs = json.load(file)

        return merge_pairs
    

    def file_exists(self, file_path):
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0


if __name__ == "__main__":
    train_set = WikiText2(split='train', encoded_text_json='data/wikitext2/encoded_text_train.json')
    val_set = WikiText2(split='val', encoded_text_json='data/wikitext2/encoded_text_val.json')
    test_set = WikiText2(split='test', encoded_text_json='data/wikitext2/encoded_text_test.json')
