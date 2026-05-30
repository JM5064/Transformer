import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from model.optimus_prime import OptimusPrime
from train import train
from model.loss import CrossEntropyLoss
from data.wikitext2 import WikiText2
from utils import DEVICE


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


    model = OptimusPrime(vocab_size=VOCAB_SIZE)
    
    loss_func = CrossEntropyLoss(vocab_size=VOCAB_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0)

    model = model.to(DEVICE)

    num_warmup_epochs = 10
    num_epochs = 100


    def convnext_scheduler(optimizer, num_warmup_epochs, total_epochs):
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/num_warmup_epochs, total_iters=num_warmup_epochs)

        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs-num_warmup_epochs, eta_min=1e-5)

        return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_epochs])
    
    scheduler = convnext_scheduler(optimizer, num_warmup_epochs, num_epochs)
    
    # start training (!!)
    train(model, num_epochs, train_loader, val_loader, test_loader, loss_func, optimizer, scheduler, 
          start_epoch=0,
          runs_dir='runs'
    )    



    