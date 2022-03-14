import numpy as np
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from utils import *
from datetime import datetime

print("Method: BALD (curiosity)")

active_learning_steps = 5 # Number of time to gather new data
active_learning_data_per_step = 1000 # Amount of data gathered per step
batch_size = 100
epochs = 10 # Number of epochs to train on per active_learning_step
print(f"Active learning steps: {active_learning_steps}")
print(f"Active learning data per step: {active_learning_data_per_step}")
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

current_train_indices = set()

init_weights = deepcopy(net.state_dict())

now = datetime.now()

for i in range(active_learning_steps):
    print(f"Data iteration {i}")
    net.train()

    with torch.no_grad():
        
        inputs = X_train
        inputs = torch.Tensor(inputs)
        outputs = net(inputs)
        
        active_learning_data_per_step = 1000
        print(f"Finding best {active_learning_data_per_step} items in pool of {len(inputs)}")
        iterations = 100

        predictions = np.zeros((inputs.shape[0], 1, iterations))

        if i > 0:
            with torch.no_grad():
                for it_idx in range(iterations):
                    output = net(inputs)
                    output = output.detach().cpu().numpy()
                    predictions[:, 0, it_idx] = output.squeeze(1)
        
        scores = compute_score(predictions)
        indices = np.argsort(scores)[::-1]

        train_idx = 0
        while len(current_train_indices) < ((i + 1)*active_learning_data_per_step):
            current_train_indices.add(indices[train_idx])
            train_idx += 1
        
        X_train_ = X_train[list(current_train_indices)]
        y_train_ = y_train[list(current_train_indices)]
        print(f"Training on {len(current_train_indices)} items")
    
    net.load_state_dict(init_weights)
    for epoch in range(epochs * i, (epochs * (i+1))):
        np.random.seed(epoch)
        np.random.shuffle(X_train_)
        np.random.seed(epoch)
        np.random.shuffle(y_train_)
        for idx in range(0, len(X_train_), batch_size):
            step = idx + (epoch) * len(X_train_)
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
        net.eval()
        train_acc = calculate_train_accuracy(torch.Tensor(X_train), y_train, net)
        net.train()
        outputs_sigmoid_sum = None
        with torch.no_grad():
            for _ in range(iterations):
                outputs = net(torch.Tensor(X_test))
                outputs_sigmoid = torch.sigmoid(outputs)
                if outputs_sigmoid_sum is None:
                    outputs_sigmoid_sum = outputs_sigmoid
                else:
                    outputs_sigmoid_sum += outputs_sigmoid
            outputs_sigmoid_sum = outputs_sigmoid_sum / iterations
            outputs_binary = outputs_sigmoid_sum.cpu().numpy().flatten() > 0.5
        dev_acc = metrics.accuracy_score(y_test, outputs_binary)
        print(f"Train accuracy: {train_acc}")
        print(f"Dev accuracy: {dev_acc}")
print(datetime.now() - now)