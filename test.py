import torch
import numpy as np
rand_dist = torch.randint(10, (8,)).unsqueeze(1)
time_offsets = torch.arange(3).unsqueeze(0)
print(rand_dist+time_offsets)  