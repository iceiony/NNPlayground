CUDA = True

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class MLPXOR(nn.Module):
    def __init__(self):
        super(MLPXOR, self).__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 1)
        # use normal distribution
        self.l1.weight.data.normal_()
        self.l2.weight.data.normal_()

    def forward(self, x):
        x = F.sigmoid(self.l1(x))
        x = F.sigmoid(self.l2(x))
        return x

data_in = torch.Tensor([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])
data_out = torch.Tensor([0, 1, 1, 0]).resize_(4, 1)

net = MLPXOR()

if CUDA and torch.cuda.is_available():
    net = net.cuda()
    data_in = data_in.cuda()
    data_out = data_out.cuda()
    print('Running on the GPU')

inputs = Variable(data_in)
labels = Variable(data_out)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=1.0)
errors = []
for i in range(1, 500):
    optimizer.zero_grad()
    loss = criterion(net(inputs), labels)
    errors.append(loss.data.select(0, 0))
    loss.backward()
    optimizer.step()

print('Train error ', loss)
plt.plot(errors)
plt.savefig('error.png')
