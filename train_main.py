from torch.utils.data.dataloader import DataLoader
from data.wikitext2 import WikiText2


if __name__ == "__main__":
    # Load dataset
    wikitext2_train = WikiText2(encoded_text_json='data/wikitext2/encoded_text_train.json')
    wikitext2_val = WikiText2(encoded_text_json='data/wikitext2/encoded_text_val.json')
    wikitext2_test = WikiText2(encoded_text_json='data/wikitext2/encoded_text_test.json')

    # Define params
    VOCAB_SIZE = wikitext2_train.get_vocab_size()
    BATCH_SIZE = 16

    # Create dataloaders
    train_loader = DataLoader(wikitext2_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(wikitext2_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    test_loader = DataLoader(wikitext2_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    



    