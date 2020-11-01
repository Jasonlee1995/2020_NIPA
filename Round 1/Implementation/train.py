import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import numpy as np


def count(output, target):
    with torch.no_grad():
        predict = torch.argmax(output, 1)
        correct = (predict == target).sum().item()
        return correct
    

def select_model(model, num_classes):
    if model == 'resnet18':
        model_ = models.resnet18(pretrained=True)
        model_.fc = nn.Linear(512, num_classes)
    elif model == 'resnet34':
        model_ = models.resnet34(pretrained=True)
        model_.fc = nn.Linear(512, num_classes)
    elif model == 'resnet50':
        model_ = models.resnet50(pretrained=True)
        model_.fc = nn.Linear(2048, num_classes)
    elif model == 'resnet101':
        model_ = models.resnet101(pretrained=True)
        model_.fc = nn.Linear(2048, num_classes)
    return model_


class Baseline():
    def __init__(self, model, num_classes, gpu_id=0, print_freq=10, epoch_print=1, save='./best.pt'):
        self.gpu = gpu_id
        self.print_freq = print_freq
        self.epoch_print = epoch_print
        self.save = save

        torch.cuda.set_device(self.gpu)

        self.loss_function = nn.CrossEntropyLoss().cuda(self.gpu)

        model = select_model(model, num_classes)
        self.model = model.cuda(self.gpu)

        self.train_losses = []
        self.train_acc = []
        self.test_losses = []
        self.test_acc = []
        self.best_acc = None


    def train(self, train_data, test_data, epochs=100, lr=0.1, weight_decay=0.0001):
        # Model to Train Mode
        self.model.train()

        # Set Optimizer and Scheduler
        optimizer = optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)

        # Train
        for epoch in range(epochs):
            if epoch % self.epoch_print == 0:
                print('Epoch {} Started...'.format(epoch+1))
            for i, (X, y) in enumerate(train_data):
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                output = self.model(X)
                loss = self.loss_function(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch % self.epoch_print == 0) and ((i+1) % self.print_freq == 0):
                    train_acc = 100 * count(output, y) / y.size(0)
                    test_acc, test_loss = self.test(test_data)
                    
                    if (self.best_acc == None) or (self.best_acc < test_acc):
                        torch.save(self.model.state_dict(), self.save)
                        self.best_acc = test_acc
                        print('Best Model Saved')

                    self.train_losses.append(loss.item())
                    self.train_acc.append(train_acc)
                    self.test_losses.append(test_loss)
                    self.test_acc.append(test_acc)

                    self.model.train()

                    print('Iteration : {} - Train Loss : {:.6f}, Test Loss : {:.6f}, '
                          'Train Acc : {:.6f}, Test Acc : {:.6f}'.format(i+1, loss.item(), test_loss, train_acc, test_acc))


    def test(self, test_data):
        correct, total = 0, 0
        losses = []

        # Model to Eval Mode
        self.model.eval()

        # Test
        with torch.no_grad():
            for i, (X, y) in enumerate(test_data):
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                output = self.model(X)

                loss = self.loss_function(output, y)
                losses.append(loss.item())
                
                correct += count(output, y)
                total += y.size(0)
                
        return (100*correct/total, sum(losses)/len(losses))