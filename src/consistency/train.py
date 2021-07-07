import copy
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as datasets

from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

from src.consistency.utils import (
    CustomScheduler, ICTDataset, SmallDataset, CifarCNN)
from src.consistency.mean_teacher import MeanTeacher


LR = 0.01
SAMPLES = 50
BATCH_SIZE = 10
ICT_BATCH_SIZE = 400
NUM_EPOCHS = 50
USE_CONSISTENCY = True
SSL_TECHNIQUE = 'ICT'
assert SSL_TECHNIQUE in {'ICT', 'mean teacher'}
SSL_WEIGHT = 1.0
SSL_WEIGHT_SCHEDULER = None # CustomScheduler(0, 5, NUM_EPOCHS, 1)
ALPHA = 0.999
ALPHA_SCHEDULER = None # CustomScheduler(0.9, 0.9999, NUM_EPOCHS, 4)

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.GaussianBlur(3),
    transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1)),
    transforms.ToTensor()
])
VAL_TRANSFORMS = transforms.ToTensor()


def train():
    # Datasets and Loaders
    trainset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=TRAIN_TRANSFORMS)
    valset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=VAL_TRANSFORMS)
    small_dataset = SmallDataset(trainset, SAMPLES)
    data_loader = DataLoader(small_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(valset, batch_size=len(valset), shuffle=False)

    # Model and losses
    model = CifarCNN()
    model.to('cuda:0')
    optimiser = torch.optim.SGD(model.parameters(), lr=LR)
    loss_f = nn.CrossEntropyLoss()

    if USE_CONSISTENCY:
        ict_dataset = ICTDataset(trainset)
        ict_data_loader = DataLoader(ict_dataset, batch_size=ICT_BATCH_SIZE, shuffle=True)
        loss_ict_f = nn.MSELoss(reduction='sum')
        teacher_model = copy.deepcopy(model)
        mean_teacher = MeanTeacher(ALPHA, teacher_model, ALPHA_SCHEDULER)
        softmax = nn.Softmax(dim=1)

    for epoch in range(NUM_EPOCHS):
        if SSL_WEIGHT_SCHEDULER:
            SSL_WEIGHT = SSL_WEIGHT_SCHEDULER(epoch)
        model.train()
        for (img_batch, labels), (ict_img_batch1, ict_img_batch2) in zip(data_loader, ict_data_loader):
            optimiser.zero_grad()
            img_batch = img_batch.cuda()
            preds = model(img_batch)
            loss = loss_f(preds, labels.cuda())
            loss.backward()
            optimiser.step()

            if not USE_CONSISTENCY:
                continue

            optimiser.zero_grad()
            if SSL_TECHNIQUE == 'ICT':
                interpolation_coef = float(torch.rand(1))
                ict_img_batch3 = ict_img_batch1 * interpolation_coef + (1.0 - interpolation_coef) * ict_img_batch2
                pred1 = mean_teacher.model(ict_img_batch1.cuda())
                pred2 = mean_teacher.model(ict_img_batch2.cuda())
                pred = model(ict_img_batch3.cuda())
                pred1 = softmax(pred1)
                pred2 = softmax(pred2)
                pred = softmax(pred)
                labels = interpolation_coef * pred1 + (1.0 - interpolation_coef) * pred2
            elif SSL_TECHNIQUE == 'mean teacher':
                labels = mean_teacher.model(ict_img_batch1.cuda())
                pred = model(ict_img_batch1.cuda())
            else:
                raise ValueError('wroong SSL technique', SSL_TECHNIQUE)
            loss = loss_ict_f(pred, labels)
            loss *= SSL_WEIGHT
            loss.backward()
            optimiser.step()
            mean_teacher.optimise(model, epoch)

        model.eval()
        for img_batch, labels in val_data_loader:
            img_batch = img_batch.cuda()
            preds = model(img_batch)
            preds = torch.max(preds, dim=1).indices
            accuracy = accuracy_score(preds.to('cpu'), labels)


if __name__=='__main__':
    train()
