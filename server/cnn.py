import torch
import torch.nn as nn
torch.manual_seed(1)
model = nn.Sequential()

model.add_module(
    'conv_1',
    nn.Conv2d(in_channels=3, out_channels=32,
             kernel_size=3, padding=1)
)
model.add_module('relu_1', nn.ReLU())
model.add_module('pool_1', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout_1', nn.Dropout(0.5))

model.add_module(
    'conv_2',
    nn.Conv2d(in_channels=32, out_channels=64,
              kernel_size=3, padding=1)
)
model.add_module('relu_2', nn.ReLU())
model.add_module('pool_2', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout_2', nn.Dropout(0.5))

model.add_module(
    'conv_3',
    nn.Conv2d(in_channels=64, out_channels=128,
              kernel_size=3, padding=1)
)
model.add_module('relu_3', nn.ReLU())
model.add_module('pool_3', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout_3', nn.Dropout(0.5))

model.add_module(
    'conv_4',
    nn.Conv2d(in_channels=128, out_channels=256,
              kernel_size=3, padding=1)
)
model.add_module('relu_4', nn.ReLU())
model.add_module('pool_4', nn.MaxPool2d(kernel_size=2))

model.add_module(
    'conv5',
    nn.Conv2d(
        in_channels=256, out_channels = 512,
        kernel_size=3, padding=1
    )
)
model.add_module('relu5', nn.ReLU())
model.add_module('pool5', nn.AvgPool2d(kernel_size=8))

model.add_module('flatten', nn.Flatten())

model.add_module('fc', nn.Linear(512, 3))