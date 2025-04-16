import torch
g = torch.Generator()
g.manual_seed(1)
print(g.initial_seed())

g = torch.Generator()
g.manual_seed(2)
print(g.initial_seed())