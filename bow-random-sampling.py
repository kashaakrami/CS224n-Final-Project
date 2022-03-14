import numpy as np
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from utils import *
from datetime import datetime

print("Method: random sampling (baseline)")

batch_size = 100
epochs = 10 # Number of epochs to train on per active_learning_step
print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")

X_train, X_test, y_train, y_test = get_dataset_df(tensor=False)
print(f'Shape of train data is {X_train.shape}')
print(f'Shape of test data is {X_test.shape}')

torch.manual_seed(0)
net = Model()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters())

losses = []
steps = []

indices = np.random.choice(len(X_train), size=5000, replace=False)
X_train_ = X_train[indices]
y_train_ = y_train[indices]

now = datetime.now()

for epoch in range(epochs):
    np.random.seed(epoch)
    np.random.shuffle(X_train_)
    np.random.seed(epoch)
    np.random.shuffle(y_train_)
    for idx in range(0, len(X_train_), batch_size):
        step = idx + epoch * len(X_train_)
        steps.append(step)

        inputs = X_train_[idx:idx+batch_size]
        inputs = torch.Tensor(inputs)
        labels = y_train_[idx:idx+batch_size]
        labels = torch.Tensor(labels)
        labels = labels.unsqueeze(1)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}")
    train_acc = calculate_train_accuracy(torch.Tensor(X_train), y_train, net)
    dev_acc = calculate_dev_accuracy(torch.Tensor(X_test), y_test, net)
    print(f"Train accuracy: {train_acc}")
    print(f"Dev accuracy: {dev_acc}")

print(datetime.now() - now)