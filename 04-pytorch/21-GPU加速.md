# GPU加速

## 1. 用GPU训练CNN

这份 GPU 的代码是依据之前这份CNN的代码修改的. 大概修改的地方包括将数据的形式变成 GPU 能读的形式, 然后将 CNN 也变成 GPU 能读的形式. 做法就是在后面加上 .cuda(), 很简单.

### 1.1 tensor数据改成GPU兼容

```py
...

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# !!!!!!!! 修改 test data 形式 !!!!!!!!! #
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
test_y = test_data.test_labels[:2000].cuda()
```

### 1.2 再来把我们的 CNN 参数也变成 GPU 兼容形式

```py
class CNN(nn.Module):
    ...

cnn = CNN()

# !!!!!!!! 转换 cnn 去 CUDA !!!!!!!!! #
cnn.cuda()      # Moves all model parameters and buffers to the GPU.
```

### 1.3 然后就是在 train 的时候, 将每次的training data 变成 GPU 形式. + .cuda()

```py
for epoch ..:
    for step, ...:
        # !!!!!!!! 这里有修改 !!!!!!!!! #
        b_x = x.cuda()    # Tensor on GPU
        b_y = y.cuda()    # Tensor on GPU

        ...

        if step % 50 == 0:
            test_output = cnn(test_x)

            # !!!!!!!! 这里有修改  !!!!!!!!! #
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # 将操作放去 GPU

            accuracy = torch.sum(pred_y == test_y) / test_y.size(0)
            ...

test_output = cnn(test_x[:10])

# !!!!!!!! 这里有修改 !!!!!!!!! #
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # 将操作放去 GPU
...
print(test_y[:10], 'real number')
```

## 2. 转移到CPU

如果你有些计算还是需要在 CPU 上进行的话呢, 比如 plt 的可视化, 我们需要将这些计算或者数据转移至 CPU.

```py
cpu_data = gpu_data.cpu()
```