import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import time

import torch
from utils import DEVICE, log_results


def validate(model, val_loader, loss_func):
    model.eval()
    total_loss= 0.0
    
    with torch.no_grad():
        for X, Y in tqdm(val_loader):
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            # Get predictions for regression and heatmap paths
            preds = model(X)
            loss = loss_func(Y, preds)
            total_loss += loss.item()


    average_val_loss = total_loss / len(val_loader)
    
    metrics = {
        "average_val_loss": average_val_loss,
    }

    return metrics


def train(
        model,
        num_epochs,
        train_loader,
        val_loader,
        test_loader,
        loss_func,
        optimizer,
        scheduler,
        start_epoch=0,
        runs_dir="runs",
    ):
    log_directory = runs_dir
    # create log file for a new training session
    if start_epoch == 0:
        time = str(datetime.now())
        os.mkdir(runs_dir + "/" + time)
        log_directory = runs_dir + "/" + time
    best_loss = float('inf')

    # training loop
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        total_loss = 0.0

        model.train()
        for X, Y in tqdm(train_loader):
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            optimizer.zero_grad()

            # Get predictions
            preds = model(X)

            loss = loss_func(Y, preds)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # print and log metrics
        average_train_loss = total_loss / len(train_loader)

        metrics = validate(model, val_loader, loss_func)
        metrics["average_train_loss"] = average_train_loss

        print(f'Epoch {epoch+1} Results:')

        print(f'Train Loss: {average_train_loss}')
        print(f'Val Loss:   {metrics["average_val_loss"]}')

        log_results(log_directory + "/metrics.csv", metrics)

        # Step scheduler
        if scheduler:
            scheduler.step()

        # save best model
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        # Save best model
        val_loss = metrics['average_val_loss']
        if val_loss < best_loss:
            torch.save(checkpoint, log_directory + "/best.pt")
            best_loss = val_loss

        # Save last model
        torch.save(checkpoint, log_directory + "/last.pt")


    # test model and print/log testing metrics
    print("Testing Model")
    metrics = validate(model, test_loader, loss_func)
    print("Testing Results")
    print(f'Test Loss:   {metrics["average_val_loss"]} ')

    test_logfile_path = log_directory + "/test_metrics.csv"
    log_results(test_logfile_path, metrics)
