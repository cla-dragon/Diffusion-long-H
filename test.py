import torch

mean = torch.zeros(2)
std = torch.ones(2)
dist = torch.distributions.Normal(mean, std)
dist = torch.distributions.Independent(dist, 1)
log_prob = dist.log_prob(torch.tensor([[0.5, -0.5], [0, 0.0]]))
print(log_prob)