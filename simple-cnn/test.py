import torch
from train import simplecnn

x = torch.randn(32, 3, 224, 224)
model = simplecnn(num_class=4) # 实例化
output = model(x)
print(output.shape)