
# torch函数

## 1. torch.cat

在给定维度上对输入的张量序列seq 进行连接操作。

```py
torch.cat(inputs, dimension=0) → Tensor

>>> x = torch.randn(2, 3)
>>> x

 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
[torch.FloatTensor of size 2x3]

>>> torch.cat((x, x, x), 0)

 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
 0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735
[torch.FloatTensor of size 6x3]

>>> torch.cat((x, x, x), 1)

 0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918
 1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735
[torch.FloatTensor of size 2x9]
```

## 2. torch.max

返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引。
输出形状中，将dim维设定为1，其它与输入形状保持一致。
返回的是个二元tuple，第一个为元素值，第二个为位置索引

```py
torch.max(input, dim, max=None, max_indices=None) -> (Tensor, LongTensor)

>> a = torch.randn(4, 4)
>> a

0.0692  0.3142  1.2513 -0.5428
0.9288  0.8552 -0.2073  0.6409
1.0695 -0.0101 -2.4507 -1.2230
0.7426 -0.7666  0.4862 -0.6628
torch.FloatTensor of size 4x4]

>>> torch.max(a, 1)
(
 1.2513
 0.9288
 1.0695
 0.7426
[torch.FloatTensor of size 4x1]
,
 2
 0
 0
 0
[torch.LongTensor of size 4x1]
)
```

## 3. save load

```py
torch.save(net1, 'net.pkl')  # 保存整个网络
torch.save(net1.state_dict(), 'net_params.pkl')   # 只保存网络中的参数

net2 = torch.load('net.pkl')
prediction = net2(x)

net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

# 将保存的参数复制到 net3
net3.load_state_dict(torch.load('net_params.pkl'))
prediction = net3(x)
```

## 4. 训练过程

```py
prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

loss = loss_func(prediction, y)     # 计算两者的误差

optimizer.zero_grad()   # 清空上一步的残余更新参数值
loss.backward()         # 误差反向传播, 计算参数更新值
optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
```

## 5. 快速搭建

```py
net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
```

## 6. Data.TensorDataset || Data.DataLoader

```py
import torch.utils.data as Data

train_data = Data.TensorDataset(data_tensor=x, target_tensor=y)
# 把x,y包装成元祖对的形式
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)
```

## 7. different optimizer

```py
# different optimizers
opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
```

## 8. tensor.view(*args) → Tensor

张量变形函数

```py
>>> x = torch.randn(4, 4)
>>> x.size()
torch.Size([4, 4])
>>> y = x.view(16)
>>> y.size()
torch.Size([16])
>>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
>>> z.size()
torch.Size([2, 8])
```

## 9. Variable tensor numpy转换

### 9.1 tensor <-> numpy

```py
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
```

### 9.2 Variable <-> tensor

```py
tensor = torch.FloatTensor([[1,2],[3,4]])
# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensor, requires_grad=True)
tensor = variable.data
```

### 9.3 tensor.data = tensor

## 10. pytorch主要层的函数

```py
from torch.nn import \
BatchNorm2d, BatchNorm3d, Conv2d, Conv3d, LeakyReLU,
Linear, MaxPool2d, Sigmoid, Tanh
```

### 10.1 卷积层Conv2d

二维卷积层, 输入的尺度是(N, C_in,H,W)，输出尺度（N,C_out,H_out,W_out）
卷积核也有通道值 卷积核的size （C_out, C_in, F, F)
卷积核的数目为out_channels，卷积核的通道数为in_channels

```py
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

+ in_channels(int) – 输入信号的通道
+ out_channels(int) – 卷积产生的通道
+ kerner_size(int or tuple) - 卷积核的尺寸
+ stride(int or tuple, optional) - 卷积步长
+ padding(int or tuple, optional) - 输入的每一条边补充0的层数
+ dilation(int or tuple, optional) – 卷积核元素之间的间距
+ groups(int, optional) – 从输入通道到输出通道的阻塞连接数
+ bias(bool, optional) - 如果bias=True，添加偏置

### 10.2 卷积层Conv3d

三维卷积层, 输入的尺度是(N, C_in,D,H,W)，输出尺度（N,C_out,D_out,H_out,W_out）

```py
class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

+ in_channels(int) – 输入信号的通道
+ out_channels(int) – 卷积产生的通道
+ kerner_size(int or tuple) - 卷积核的尺寸
+ stride(int or tuple, optional) - 卷积步长
+ padding(int or tuple, optional) - 输入的每一条边补充0的层数
+ dilation(int or tuple, optional) – 卷积核元素之间的间距
+ groups(int, optional) – 从输入通道到输出通道的阻塞连接数
+ bias(bool, optional) - 如果bias=True，添加偏置

### 10.3 池化层MaxPool2d

对于输入信号的输入通道，提供2维最大池化（max pooling）操作

```py
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

+ kernel_size(int or tuple) - max pooling的窗口大小
+ stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
+ padding(int or tuple, optional) - 输入的每一条边补充0的层数
+ dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参
+ return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
+ ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

### 10.4 激活层LeakyReLU

```py
class torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
```

对输入的每一个元素运用$f(x) = max(0, x) + {negative_slope} * min(0, x)$

### 10.5 激活层Sigmod Tanh

```py
class torch.nn.Sigmoid [source]
```

对每个元素运用Sigmoid函数，Sigmoid 定义如下：

```math
    f(x)=1/(1+e^{−x})
```

### 10.6 激活层Tanh

```py
class torch.nn.Tanh [source]
```

对输入的每个元素，

```math
f(x)=e^x−e^{−x}/e^x+e^{-x}
```

### 10.7 全连接层Linear

```py
class torch.nn.Linear(in_features, out_features, bias=True)
```

对输入数据做线性变换：y=Ax+b

+ n_features - 每个输入样本的大小
+ out_features - 每个输出样本的大小
+ bias - 若设置为False，这层不会学习偏置。默认值：True

### 10.8 随机丢失层Dropout

```py
class torch.nn.Dropout(p=0.5, inplace=False)
```

随机将输入张量中部分元素设置为0。对于每次前向调用，被置0的元素都是随机的。

+ p - 将元素置0的概率。默认值：0.5
+ in-place - 若设置为True，会在原地执行操作。默认值：False

### 10.9 BN层BatchNorm2d

```py
class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)[source]
```

对小批量(mini-batch)3d数据组成的4d输入进行批标准化(Batch Normalization)操作

```math
y = \frac{x - mean[x]}{ \sqrt{Var[x]} + \epsilon} * gamma + beta
```

在每一个小批量（mini-batch）数据中，计算输入各个维度的均值和标准差。gamma与beta是可学习的大小为C的参数向量（C为输入大小）
在训练时，该层计算每次输入的均值与方差，并进行移动平均。移动平均默认的动量值为0.1。

在验证时，训练求得的均值/方差将用于标准化验证数据。

+ num_features： 来自期望输入的特征数，该期望输入的大小为'batch_size x num_features x height x width'
+ eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
+ momentum： 动态均值和动态方差所使用的动量。默认为0.1。
+ affine： 一个布尔值，当设为true，给该层添加可学习的仿射变换参数。

### 10.10 BN层BatchNorm3d

```py
class torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True)
```

对小批量(mini-batch)4d数据组成的5d输入进行批标准化(Batch Normalization)操作

+ num_features： 来自期望输入的特征数，该期望输入的大小为'batch_size x num_features x depth x height x width'
+ eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
+ momentum： 动态均值和动态方差所使用的动量。默认为0.1。
+ affine： 一个布尔值，当设为true，给该层添加可学习的仿射变换参数。