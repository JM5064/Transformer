import bpe
from torch.utils.data import Dataset


class WikiText2(Dataset):

    def __init__(self, seq_len=128, 
            encoded_text_json='data/wikitext2/encoded_text.json',
            vocab_json='data/wikitext2/vocab.json', 
        ):

        super().__init__()

        self.seq_len = seq_len

        # Load data files
        self.encoded_text = bpe.load_from_file(encoded_text_json)
        self.vocab = bpe.load_from_file(vocab_json)


    def __getitem__(self, index):
        """Retrives a training + target sample given an index
            The training sample is a piece of text start at index 'index' of length seq_len
            The target is the same piece of text, shifted right by 1
        """

        input_tokens = self.encoded_text[index : index + self.seq_len]
        target_tokens = self.encoded_text[index + 1 : index + self.seq_len + 1]

        return input_tokens, target_tokens
    

    def __len__(self):
        return len(self.encoded_text) - self.seq_len
    

if __name__ == "__main__":
    train_set = WikiText2(encoded_text_json='data/wikitext2/encoded_text_train.json')
    val_set = WikiText2(encoded_text_json='data/wikitext2/encoded_text_val.json')
    test_set = WikiText2(encoded_text_json='data/wikitext2/encoded_text_test.json')
