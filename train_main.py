import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from model.bumblebee.bumblebee import Bumblebee
from train import train
from model.loss import CrossEntropyLoss
from data.wikitext2 import WikiText2
from utils import DEVICE


if __name__ == "__main__":
    # Load dataset
    wikitext2_train = WikiText2(encoded_text_json='data/wikitext2/encoded_text_train.json')
    wikitext2_val = WikiText2(encoded_text_json='data/wikitext2/encoded_text_val.json', percent=0.1)
    wikitext2_test = WikiText2(encoded_text_json='data/wikitext2/encoded_text_test.json')

    # # FOR TESTING
    # wikitext2_val = WikiText2(encoded_text_json='data/wikitext2/encoded_text_mini.json')

    # Define params
    VOCAB_SIZE = wikitext2_train.get_vocab_size()
    BATCH_SIZE = 32

    # Create dataloaders
    train_loader = DataLoader(wikitext2_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(wikitext2_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(wikitext2_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    d_model = 512
    model = Bumblebee(vocab_size=VOCAB_SIZE, d_model=d_model)
    
    loss_func = CrossEntropyLoss(vocab_size=VOCAB_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0)

    model = model.to(DEVICE)

    num_warmup_steps = 4000


    def update_learning_rate(optimizer, step_num):
        new_lr = d_model ** -0.5 * min(step_num ** -0.5, step_num * num_warmup_steps ** -1.5)

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    
    # start training (!!)
    train(model, train_loader, val_loader, test_loader, loss_func, optimizer, update_learning_rate, 
          start_epoch=0,
          runs_dir='runs'
    )    



    