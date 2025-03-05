import matplotlib.pyplot as plt

import torch
import numpy as np


thresh_drop = 0.9
thresh_quant = 0.8

data = torch.load('./tests/measurement_result.pt')

table = torch.stack(data['table']).numpy()

std = np.std(table, axis=0)
mean = np.mean(table, axis=0)
print(std)
print(mean)

sorted_ixs = sorted(range(len(mean)), key=lambda ix: mean[ix], reverse=True)
print(sorted_ixs)

status = [2] * len(mean)

total = mean.sum()
grand_total = 0
for ix in sorted_ixs:
    grand_total += mean[ix]
    rate = grand_total / total
    if rate > thresh_drop:
        print('drop id:', ix, rate)
        status[ix] = 0
    elif rate > thresh_quant:
        print('quant id:', ix, rate)
        status[ix] = 1
print(status)

plt.subplot(2, 1, 1)
plt.bar(range(len(std)), std)
plt.subplot(2, 1, 2)
plt.bar(range(len(mean)), mean, color='orange')
plt.show()
