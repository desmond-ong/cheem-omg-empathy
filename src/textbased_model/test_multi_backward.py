'''
https://discuss.pytorch.org/t/how-to-use-the-backward-functions-for-multiple-losses/1826/7
Simon Wang

accumulate the loss --> backward --> step()
is the SAME with:
loss --> backward --> next loss --> backward .... --> step
'''

from torch import nn
from torch.autograd import Variable
import torch
import torch.optim as optim


l1 = nn.Linear(3, 3)
l1.weight.data.fill_(0)
l1.bias.data.fill_(0)
opt = optim.SGD(l1.parameters(), lr=0.001)

x = Variable(torch.ones(2, 3))

# backward one loss only
loss1 = (l1(x) - 1).abs().sum()
loss1.backward()
opt.step()
print(l1.weight)
#
w1 = l1.weight.grad
print(w1)


print("--------------")
# l1.weight.grad = None
l1.zero_grad()
loss2 = (2 * l1(x) + 3).abs().sum()
loss2.backward()
opt.step()
print(l1.weight)

w2 = l1.weight.grad
print(w2)


print("--------------")
# l1.weight.grad = None
loss1 = (l1(x) - 1).abs().sum()
loss2 = (2 * l1(x) + 3).abs().sum()
(loss1+loss2).backward()
print(l1.weight.grad)
opt.step()
print(l1.weight)

# w12 = l1.weight.grad
# print(w12)


print("--------------")
l1.weight.grad = None
loss1 = (l1(x) - 1).abs().sum()
loss1.backward()
loss2 = (2 * l1(x) + 3).abs().sum()
loss2.backward()
w12_2 = l1.weight.grad
print(w12_2)
print(l1.weight)
