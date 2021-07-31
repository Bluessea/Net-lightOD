import torch
import torch.nn as nn

input = torch.randn(1,416,416)

avgpool1 = nn.AvgPool1d(3,stride=2,padding=1)

avgpool2 = nn.AvgPool2d(3,stride=2)

out1 = avgpool1(input)
out2 = avgpool2(input)
print(out1)