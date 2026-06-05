import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from model.bumblebee.bumblebee import Bumblebee
from train import validate
from model.loss import CrossEntropyLoss
from data.wikitext2 import WikiText2
from utils import DEVICE


if __name__ == "__main__":
    # Load dataset
    wikitext2_test = WikiText2(encoded_text_json='data/wikitext2/encoded_text_test.json')

    # # FOR TESTING
    # wikitext2_val = WikiText2(encoded_text_json='data/wikitext2/encoded_text_mini.json')

    # Define params
    VOCAB_SIZE = wikitext2_test.get_vocab_size()
    BATCH_SIZE = 32

    # Create dataloaders
    test_loader = DataLoader(wikitext2_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    d_model = 512
    model = Bumblebee(vocab_size=VOCAB_SIZE, d_model=d_model)
    
    loss_func = CrossEntropyLoss(vocab_size=VOCAB_SIZE, label_smoothing=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0)

    model = model.to(DEVICE)

    # Modify model path for desired model
    MODEL_PATH = 'runs/every128,attentiondropout/best.pt'

    model_state_dict = torch.load(MODEL_PATH)['state_dict']
    model.load_state_dict(model_state_dict)

    print("Testing model...")
    metrics = validate(model, test_loader, CrossEntropyLoss(vocab_size=VOCAB_SIZE, label_smoothing=0.0))
    print(metrics)



    