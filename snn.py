# snnTorch Library
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

# PyTorch Library
import torch
import torch.optim as optim
import torch.nn as nn # PyTorch neural network
from torch.utils.data import DataLoader # data loader
from torchvision import datasets, transforms # CV

# Other Libraries
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------ #


# Nueral Networks Parameters
batch_size = 128
num_inputs = 28*28 # 784 input neurons
num_hidden = 300
num_outputs = 10

epochs = 1 # 训练轮次
loss_hist = [] # record train loss
test_loss_hist = [] # record test loss
counter = 0 # use for batch counter

# Leaky Integrate-and-Fire (LIF) Parameters
beta = 0.95
num_steps = 25

# ------------------------------ #

def print_batch_accuracy(data, targets, train=False):
    output, _ = snnModel(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {e}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

# ------------------------------ #

# Check Device Availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print('Device:', device) # mps: metal programming system (on macos)

# Define the Transform (Data Pre-processing)
transform = transforms.Compose([
            transforms.Resize((28, 28)), # resize image
            transforms.Grayscale(), # RGB (3 channels) to greyscale (1 channel)
            transforms.ToTensor(), # image to tensor
            transforms.Normalize((0,), (1,))]) # unified data format [0, 1]

# MNIST Dataset
data_path = '/tmp/data/mnist' # data path
dtype = torch.float # tensor data type (float32)

# Create and Load the MNIST Dataset
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

print(f"The size of mnist_train is {len(mnist_train)}")
print(f"The size of mnist_test is {len(mnist_test)}")

# Create Dataloaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, 
                         batch_size=batch_size, 
                         shuffle=True, # shuffle the data (avoid data order bias)
                         drop_last=True) # drop the last incomplete batch (not enough 128 samples)

# ------------------------------ #

# Spiking Neural Networks
class spikingNet(nn.Module):
    def __init__(self):
        super().__init__() # 确保 spikingNet (subclass) 能够继承 nn.Module (parent class) 的属性和方法

        # Neural Networks Layer Structure (3 layers)
        self.fcl1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta) # (define in __init__ to) store/record the parameters
        self.fcl2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
    
    def forward(self, x): # forward pass neural networks
        memV1 = self.lif1.reset_mem() # membrane V reset
        memV2 = self.lif2.reset_mem()

        spike_recorded = [] # output/final layer spike record
        memV_recorded = []

        for s in range(num_steps): # process 'num_steps' time steps
            cur1 = self.fcl1(x)
            spike1, memV1 = self.lif1(cur1, memV1) # first layer computing, by LIF
            cur2 = self.fcl2(spike1)
            spike2, memV2 = self.lif2(cur2, memV2)
            spike_recorded.append(spike2)
            memV_recorded.append(memV2)
        
        return torch.stack(spike_recorded, dim=0), torch.stack(memV_recorded, dim=0)

snnModel = spikingNet().to(device) # model initialisation (move the model to device)

# ------------------------------ #

# Define the loss function and the optimiser
criterion = nn.CrossEntropyLoss() # loss function (softmax layer)
optimiser = optim.Adam(snnModel.parameters(), # Adam optimiser
                       lr=0.001, # lr: learning rate (每次参数更新的步长)
                       betas=(0.9, 0.999)) # beats: 衰减率

# ------------------------------ #

for e in range(epochs):
    iter_counter = 0 # initialise the iterator
    train_batch = iter(train_loader) # iterate and load the train data

    # Batch Iteration
    for data, targets in train_batch: # data 数据, targets 数据所对应的标签 (0-9)
        data = data.to(device)
        targets = targets.to(device)

        # --- Training Process --- #
        
        # Forward Pass
        snnModel.train() # set model to the training mode
        spike_rec, memV_rec = snnModel(data.view(batch_size, -1)) # flatten the tensor and keep the batch structure

        # Loss Computing
        loss_val = torch.zeros((1), dtype = dtype, device = device) # intialise loss value

        # 累加每一个 timestep 的 loss
        for step in range(num_steps):
            loss_val += criterion(memV_rec[step], targets)
        
        # Backward Pass (weight update) 
        optimiser.zero_grad() # 清除梯度
                              # clear the previous calculated/stored weights/gradients, 
                              # to make sure the backward() calculation correspondng to current batch
        loss_val.backward() # 计算梯度 (反向传播, 计算损失函数对模型参数的梯度)
                            # computes the gradient of current tensor
        optimiser.step() # 根据梯度更新/调整参数
                         # update the weights/gradients (update the model)
        
        loss_hist.append(loss_val.item()) # record the loss of current batch

        # --- Testing Process --- #
        
        with torch.no_grad(): # close the gradient calculation
            snnModel.eval() # set to evaluation mode
            test_data, test_targets = next(iter(test_loader)) # take one batch from test data set
            
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = snnModel(test_data.view(batch_size, -1)) # flatten, and get the spiking record and memV record

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device) # initialise the test loss value

            for step in range(num_steps):
                test_loss += criterion(test_mem[step], test_targets)

            test_loss_hist.append(test_loss.item()) # record the test loss

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer()
            counter += 1
            iter_counter +=1

# ------------------------------ #

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist, lw=0.25)
plt.plot(test_loss_hist, lw=0.25)
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig("train_and_loss.png")

# ------------------------------ #

total = 0 # initialise total samples number
correct = 0 # initialise correct samples number

# drop_last switched to False to keep all samples
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
  snnModel.eval()
  for data, targets in test_loader:
    data = data.to(device)
    targets = targets.to(device)
    
    test_spk, _ = snnModel(data.view(data.size(0), -1)) # forward pass
    _, predicted = test_spk.sum(dim=0).max(1) # calculate total accuracy

    total += targets.size(0)
    correct += (predicted == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

# ------------------------------ #

test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

with torch.no_grad():
  snnModel.eval()
  for data, targets in test_loader:
    data = data.to(device)
    targets = targets.to(device)
    
    test_spk, _ = snnModel(data.view(data.size(0), -1)) # forward pass
    _, predicted = test_spk.sum(dim=0).max(1) # calculate total accuracy

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(data[i].cpu().numpy().squeeze(), cmap='gray')
    axes[i].set_title(f'Pred: {predicted[i].item()}')
    axes[i].axis('off')
plt.savefig("train_samples.png")

# Print true vs. predicted labels
# print("True labels: ", targets[:10].tolist())  
# print("Predicted labels: ", predicted[:10].tolist())

print("True labels: ", targets.cpu().numpy()[:10].tolist())  
print("Predicted labels: ", predicted.cpu().numpy()[:10].tolist())