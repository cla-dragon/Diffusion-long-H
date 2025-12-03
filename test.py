import torch

horizon = 10
obs_dim = 3
prior_template = torch.zeros((1, horizon, obs_dim))
prior_template[:,0] = 1
prior_template[:,-1] = 2
print(prior_template)
