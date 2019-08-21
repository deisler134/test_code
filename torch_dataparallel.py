'''
Created on Aug. 19, 2019

@author: Deisler

    example for dataparallel training with torch
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

print( torch.cuda.device_count())

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

class Model_parallel(Model):
    def __init__(self, input_size, output_size):
        super(Model_parallel, self).__init__(input_size, output_size)
        self.model = Model(input_size, output_size)
        if torch.cuda.device_count() >= 1:
            print("partion data!")
            
            self.model = nn.DataParallel(self.model)
    def forward(self, input):
        output = self.model(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
    
# model = Model(input_size, output_size)
# if torch.cuda.device_count() >= 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = nn.DataParallel(model)
# 
device = torch.device("cuda:0")
# 
# model.to(device)

model_parallel = Model_parallel(input_size, output_size)
model_parallel.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model_parallel(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
    
def parallelize_model():
    model = Model(input_size, output_size)
    if torch.cuda.device_count() >= 1:
        print("partion data function!")
        model = nn.DataParallel(model)
    return model

para_model = parallelize_model()
para_model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model_parallel(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())


    