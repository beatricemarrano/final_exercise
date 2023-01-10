import argparse
import sys
import os
import torch
import wandb
#import logging
#from pathlib import Path
#import pytorch_lightning as pl
#from pytorch_lightning import Trainer

sys.path.append("/Users/mac/Documents/GitHub/final_exercise/src")
from data.data import CorruptMnist
#sys.path.append("/Users/mac/Documents/GitHub/final_exercise/src/models")
from models.model import MyAwesomeModel

import matplotlib.pyplot as plt

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python train_model.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
           
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        
        #WANDB
        wandb.init(project="final_exercise", entity="beatrice-marrano")
        wandb.config= {'learning_rate': 1e-3, 'n_epoch': 5, 'batch_size':128}
        
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=wandb.config["learning_rate"])
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set = CorruptMnist(train=True)
        dataloader = torch.utils.data.DataLoader(train_set, batch_size=wandb.config["batch_size"])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        n_epoch = wandb.config["n_epoch"]
        for epoch in range(n_epoch):
            loss_tracker = []
            for batch in dataloader:
                optimizer.zero_grad()
                x, y = batch
                preds = model(x.to(self.device))
                loss = criterion(preds, y.to(self.device))
                loss.backward()
                optimizer.step()
                loss_tracker.append(loss.item())
                wandb.log({"Loss": loss})
            #print(loss)
            #print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss} ")
        torch.save(model, os.path.join('/Users/mac/Documents/GitHub/final_exercise/models', "trained_model.pt"))    
        torch.save(model.state_dict(), os.path.join('/Users/mac/Documents/GitHub/final_exercise/models', "checkpoint.pth"))        
            
        plt.plot(loss_tracker, '-')
        plt.xlabel('Training step')
        plt.ylabel('Training loss')
        plt.savefig(os.path.join('/Users/mac/Documents/GitHub/final_exercise/reports/figures', "training_curve.png"))
        
        return model
            
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        #WANDB
        wandb.init(project="final_exercise", entity="beatrice-marrano")
        wandb.config= {'learning_rate': 1e-3, 'n_epoch': 5, 'batch_size':128}
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        model = model.to(self.device)

        test_set = CorruptMnist(train=False)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=wandb.config["batch_size"])
        
        correct, total = 0, 0
        for batch in dataloader:
            x, y = batch
            
            preds = model(x.to(self.device))
            preds = preds.argmax(dim=-1)
            
            correct += (preds == y.to(self.device)).sum().item()
            total += y.numel()
            
            wandb.log({"Test set accuracy": correct/total})
        #print("Test set accuracy", {correct/total})


if __name__ == '__main__':
    TrainOREvaluate()

    
  
    
