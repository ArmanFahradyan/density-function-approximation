import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np


def step_function(x):
    return (x >= 0) * 1.0


# def criterion(outputs, labels, nn_outputs_1, nn_outputs_2, lambda_):
#     # model_log_probs = F.log_softmax(outputs, dim=0)
#
#     kl_loss = nn.KLDivLoss()  # Batchmean reduction is typically used
#
#     loss = kl_loss(outputs, labels)
#
#     return (1/len(labels)) * loss  # + lambda_ * (1/len(nn_outputs_1)) * torch.sum(step_function(nn_outputs_1 - nn_outputs_2) * (nn_outputs_1 - nn_outputs_2)**2)


def criterion_ascending(nn_outputs_1, nn_outputs_2):
    return (1/len(nn_outputs_1)) * torch.sum(step_function(nn_outputs_1 - nn_outputs_2) * (nn_outputs_1 - nn_outputs_2)**2)


def criterion(outputs, labels, nn_outputs_1, nn_outputs_2, lambda_):
    return nn.MSELoss()(outputs, labels) + lambda_ * criterion_ascending(nn_outputs_1, nn_outputs_2)


def criterion_ks(outputs, labels, nn_outputs_1, nn_outputs_2, lambda_):
    pass


class CorrelationLoss(nn.Module):
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, outputs, quantiles):
        mean_1 = torch.mean(outputs)
        mean_2 = torch.mean(quantiles)
        numerator = (outputs - mean_1) @ (quantiles - mean_2)
        denominator = torch.std(outputs) * torch.std(quantiles)
        if denominator < 1e-6:
            return 0.0
        return - numerator / denominator


def criterion_corr(outputs, quantiles, nn_outputs_1, nn_outputs_2, lambda_):
    return CorrelationLoss()(outputs, quantiles) + lambda_ * criterion_ascending(nn_outputs_1, nn_outputs_2)


def get_model_derivative(model):
    w1 = model.first_layer.state_dict()['0.weight']
    # b1 = model.first_layer.state_dict()['0.bias']
    w2 = model.hidden_layers[0].state_dict()['0.weight']
    # b2 = model.hidden_layers[0].state_dict()['0.bias']
    w3 = model.hidden_layers[1].state_dict()['0.weight']
    # b3 = model.hidden_layers[1].state_dict()['0.bias']
    w4 = model.last_layer.state_dict()['0.weight']
    # b4 = model.last_layer.state_dict()['0.bias']

    def derivative(x):
        x = np.array(x)
        x = x.reshape(-1, 1)
        x = torch.tensor(x, dtype=torch.float32)
        out, middle_outs = model(x)
        out = out.view(-1, 1)
        alpha_1 = 1 - middle_outs[0]**2
        alpha_2 = 1 - middle_outs[1]**2
        alpha_3 = 1 - middle_outs[2]**2
        # print("w1:", w1.shape)
        # print("w2:", w2.shape)
        # print("w3:", w3.shape)
        # print("w1:", w4.shape)
        # print("alpha_1:", alpha_1.shape)
        # print("alpha_2:", alpha_2.shape)
        # print("alpha_3:", alpha_3.shape)
        # input()
        # part1 = alpha_1 * w1.squeeze(-1)
        # part2 = w2 @ (alpha_1 * w1.squeeze(-1)).T
        # part3 = w3 @ (alpha_2 * (w2 @ (alpha_1 * w1.squeeze(-1)).T).T).T
        # part4 = alpha_3.T * (w3 @ (alpha_2 * (w2 @ (alpha_1 * w1.squeeze(-1)).T).T).T )
        # print(part4.shape)
        # print("done so far")
        # print(out.shape)
        # print((out*(1 - out)).shape)
        # input()
        ans = out*(1 - out) * (w4 @ (alpha_3.T * (w3 @ (alpha_2 * (w2 @ (alpha_1 * w1.squeeze(-1)).T).T).T ))).T
        return ans.detach().numpy()
    return derivative
