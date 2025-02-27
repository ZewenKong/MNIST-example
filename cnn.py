import torch
import torch.nn as nn

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_path = '/tmp/data/mnist'

test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

class o1Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcl1 = nn.Linear(28*28, 256)
        self.fcl2 = nn.Linear(256, 128) 
        self.fcl3 = nn.Linear(128, 64)
        self.fcl4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim = 1) 
        x = torch.relu(self.fcl1(x))
        x = torch.relu(self.fcl2(x))
        x = torch.relu(self.fcl3(x))
        x = self.fcl4(x)
        return x
    
model = o1Net()
best_model_path = './best_mnist_model.pth'
model.load_state_dict(torch.load(best_model_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Function (get 10 images from test_loader)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data, example_targets = example_data.to(device), example_targets.to(device)

# Prediction
with torch.no_grad():
    output = model(example_data)
    _, predicted = torch.max(output, 1)

# Sub-plot
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(example_data[i].cpu().numpy().squeeze(), cmap='gray')
    axes[i].set_title(f'Predicted: {predicted[i].item()}')
    axes[i].axis('off')
plt.show()

print("True labels: ", example_targets.cpu().numpy())
print("Predicted labels: ", predicted.cpu().numpy())
