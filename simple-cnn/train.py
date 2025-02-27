import torch
import torch.nn as nn

class simplecnn(nn.Module):
    def __init__(self, num_class): # num_class, 分类数
        super().__init__()
       
        self.features = nn.Sequential(  # feature extraction
            # convert 3 channels to 16 channels, and keep the size
            # 16*224*224
            nn.Conv2d(3, # input channel no.
                      16, # output channel no.
                      kernel_size=3, 
                      stride=1, # 步长
                      padding=1), # padding = 1, 保持尺寸不变
            nn.ReLU(), # 激活函数(add non-linearity features)
            # 池化: 图像大小减半, channel 数不变 (16*112*112)
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            # keep the size: 32*112*112
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 32*56*56
        ) 
        # Fully connected layer (FCL): classification
        self.classifier = nn.Sequential(
            
            nn.Linear(32*56*56, 128), # input size: 32*56*56, output size: 128
            nn.ReLU(),
            nn.Linear(128, num_class) # input size: 128, output size: num_class (分类数)
        )
    
    def forward(self, x): # x is the input data (featured image)
        x = self.features(x) # features extraction
        # FCL can only accept 1D tensor (need to flatten the 3D tensor)
        x = x.view(x.size(0), -1) 
        # '-1' means 32 3 56 56 => (32, 3*56*56)
        x = self.classifier(x) # classification
        return x 
    
# x = torch.randn(32, 3, 224, 224)
# model = simplecnn(num_class=4) # 实例化
# output = model(x)
# print(output.shape)