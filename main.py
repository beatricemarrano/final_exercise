import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel

#
import torchvision
import matplotlib.pyplot as plt


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    epochs = 5
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            # Flatten FMNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            optimizer.zero_grad()
        
            # Training pass
            logits = model(images)
        
            loss = criterion(logits, labels)
        
            loss.backward()
        
            optimizer.step()
        
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(train_set)}")
    
    torch.save(model.state_dict(), 'model_checkpoint')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()

    criterion = nn.NLLLoss()
    accuracy = 0
    test_loss = 0
    for images, labels in test_set:

        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    
    
    
    
cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    