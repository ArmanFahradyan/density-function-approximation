import torch.nn as nn


def step_function(x):
    return (x >= 0) * 1.0

# def criterion(outputs, labels, nn_outputs_1, nn_outputs_2, lambda_):
#   model_log_probs = F.log_softmax(outputs, dim=0)

#   kl_loss = nn.KLDivLoss()  # Batchmean reduction is typically used

#   loss = kl_loss(model_log_probs, labels)

#   return (1/len(labels)) * loss # + lambda_ * (1/len(nn_outputs_1)) * torch.sum(step_function(nn_outputs_1 - nn_outputs_2) * (nn_outputs_1 - nn_outputs_2)**2)


def criterion(outputs, labels, nn_outputs_1, nn_outputs_2, lambda_):
    # return torch.sum((outputs - labels)**2) + lambda_ * torch.sum(step_function(nn_outputs_1 - nn_outputs_2) * (nn_outputs_1 - nn_outputs_2)**2)
    return nn.MSELoss()(outputs, labels) # + lambda_ * (1/len(nn_outputs_1)) * torch.sum(step_function(nn_outputs_1 - nn_outputs_2) * (nn_outputs_1 - nn_outputs_2)**2)