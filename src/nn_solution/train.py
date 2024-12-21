import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from copy import deepcopy

from src.nn_solution.utils import criterion, criterion_corr


def train(model, X, delta=0.1, lambda_=0.2, num_epochs=50_000, lr=1e-3, a=-3, b=3):

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_values = []

    best_model = deepcopy(model)
    best_loss = 10**5

    for epoch in range(num_epochs):
        if epoch == 10000:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        labels = torch.rand(X.shape)
        labels, _ = torch.sort(labels, dim=0)
        # quantiles = torch.linspace(0.0, 1.0, X.shape[0])

        torch.set_printoptions(sci_mode=False)

        # N_y = 10**3
        # y = torch.FloatTensor(N_y).uniform_(a, b).view(N_y, 1)
        # nn_outputs_1, _ = model(y)
        # nn_outputs_2, _ = model(y + delta)

        running_loss = 0.0
        # Forward pass
        outputs, _ = model(X)
        loss = criterion(outputs, labels, None, None, lambda_)
        # loss = criterion_corr(outputs.flatten(), labels, nn_outputs_1, nn_outputs_2, lambda_)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        loss_values.append(running_loss)

        if running_loss < best_loss:
            best_loss = running_loss
            best_model = deepcopy(model)

    # plt.plot(np.arange(len(loss_values)), loss_values)
    # plt.show()
    # print("Training complete!")
    # print("Best Loss:", best_loss)
    return best_model
