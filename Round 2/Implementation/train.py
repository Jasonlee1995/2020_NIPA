import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import numpy as np
from sklearn.metrics import f1_score


def evaluate(predict_labels, gt_labels):
    return f1_score(predict_labels, gt_labels, average='macro')


def select_model(num_input, num_classes):
    img_features = models.resnet18(pretrained=True)
    img_features.fc = nn.Linear(512, num_classes)
    meta_features = nn.Linear(num_input, num_classes)

    classifier = nn.Linear(2*num_classes, num_classes)

    return (img_features, meta_features, classifier)


class Baseline():
    def __init__(self, num_input, num_classes, gpu_id=0, print_freq=10, epoch_print=1, save='./best'):
        self.gpu = gpu_id
        self.print_freq = print_freq
        self.epoch_print = epoch_print
        self.save = save

        torch.cuda.set_device(self.gpu)

        self.loss_function = nn.CrossEntropyLoss().cuda(self.gpu)

        img_features, meta_features, classifier = select_model(num_input, num_classes)
        self.img_features = img_features.cuda(self.gpu)
        self.meta_features = meta_features.cuda(self.gpu)
        self.classifier = classifier.cuda(self.gpu)

        self.train_losses = []
        self.val_losses = []

        self.train_f1 = []
        self.val_f1 = []

        self.best_f1 = 0


    def train(self, train_data, val_data, epochs=100, lr=0.1, weight_decay=0.0001):
        # Model to Train Mode
        self.img_features.train()
        self.meta_features.train()
        self.classifier.train()

        # Set Optimizer and Scheduler
        optimizer = torch.optim.Adam(
            [{'params': self.img_features.parameters()},
             {'params': self.meta_features.parameters(), 'lr':0.01},
             {'params': self.classifier.parameters(), 'lr':0.01}],
            lr, weight_decay=weight_decay)

        # Train
        for epoch in range(epochs):
            if epoch % self.epoch_print == 0:
                print('Epoch {} Started...'.format(epoch+1))
            for i, (X, M, y) in enumerate(train_data):
                X, M, y = X.cuda(self.gpu, non_blocking=True), M.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                img_output = self.img_features(X)
                meta_output = self.meta_features(M)
                output = self.classifier(torch.cat((img_output, meta_output), 1))

                loss = self.loss_function(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch % self.epoch_print == 0) and ((i+1) % self.print_freq == 0):
                    train_f1 = evaluate(torch.argmax(output, 1).tolist(), y.tolist())
                    val_f1, val_loss = self.val(val_data)

                    if self.best_f1 < val_f1:
                        torch.save(self.img_features.state_dict(), self.save + str(epoch) + '_' + str(i) + '_img.pt')
                        torch.save(self.meta_features.state_dict(), self.save + str(epoch) + '_' + str(i) + '_meta.pt')
                        torch.save(self.classifier.state_dict(), self.save + str(epoch) + '_' + str(i) + '_classifier.pt')
                        self.best_f1 = val_f1
                        print('Best Model Saved')

                    self.train_losses.append(loss.item())
                    self.val_losses.append(val_loss)

                    self.train_f1.append(train_f1)
                    self.val_f1.append(val_f1)

                    self.img_features.train()
                    self.meta_features.train()
                    self.classifier.train()

                    print('Iteration : {} - Train Loss : {:.6f}, Val Loss : {:.6f}, '
                          'Train F1 : {:.6f}, Val F1 : {:.6f}'.format(i+1, loss.item(), val_loss, train_f1, val_f1))


    def val(self, val_data):
        p_labels, g_labels = [], []
        losses = []

        # Model to Eval Mode
        self.img_features.eval()
        self.meta_features.eval()
        self.classifier.eval()

        # Test
        with torch.no_grad():
            for i, (X, M, y) in enumerate(val_data):
                X, M, y = X.cuda(self.gpu, non_blocking=True), M.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                img_output = self.img_features(X)
                meta_output = self.meta_features(M)
                output = self.classifier(torch.cat((img_output, meta_output), 1))

                loss = self.loss_function(output, y)
                losses.append(loss.item())

                p_labels += torch.argmax(output, 1).tolist()
                g_labels += y.tolist()

        return (evaluate(p_labels, g_labels), sum(losses)/len(losses))
