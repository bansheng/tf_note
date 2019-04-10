# Linear regression

## 1. 定义层结构

```py
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation_function=None):
        self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))
        self.b = theano.shared(np.zeros((out_size, )) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)
```

## 2. 数据构造

```py
# Make up some fake data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise        # y = x^2 - 0.5 + whitenoise

# 散点图显示
# show the fake data
plt.scatter(x_data, y_data)
plt.show()

```

## 3. 搭建神经网络

定义神经网络的输入和目标

```py
# define the inputs dtype
x = T.dmatrix("x")
y = T.dmatrix("y")
```

这里，`T.dmatrix`意味着我使用的是float64的数据类型, 和我的输入数据一致。

接着我们设计我们的神经网络，它包括两层，构成1-10-1的结构。 对于l1我们的`input_size`要和我们的x_data一致，然后我们假设了该层有10个神经元，并且以relu作为激活函数。 所以，l2以l1.output为输入，同时呢，它的输出为1维，和我们的y_data保持一致，作为神经网络的输出层，我们采用默认的线性激活函数。

```py
# determine the inputs dtype
# add layers
l1 = Layer(x, 1, 10, T.nnet.relu)
l2 = Layer(l1.outputs, 10, 1, None)
```

然后，我们定义一个cost，也就是损失函数`（cost/loss function）`，我们采用的是`l2.outputs`神经网络输出与目标值y的的平均平方差。

```py
# compute the cost
cost = T.mean(T.square(l2.outputs - y))
```

根据 cost 我们可以计算我们神经网络权值和偏置值的梯度（gradient）, 这里Theano已经集成好了对应的函数：

```py
# compute the gradients
gW1, gb1, gW2, gb2 = T.grad(cost, [l1.W, l1.b, l2.W, l2.b])
```

有了以上的基本运算步骤后，我们就可以开始，利用梯度下降法训练我们的神经网络。 首先我们定义一个学习率`learning_rate`, 这个学习率的取值一般是根据数据及实验的经验估计的，它会对模型的收敛性造成一定的影响，一般倾向于采用较小的数值。

然后，我们定义train这个函数来描述我们的训练过程，首先我们定义了函数的输入`inputs=[x, y]`, 函数的输出为`outputs=cost`, 同时更新网络的参数

```py
# apply gradient descent
learning_rate = 0.05
train = theano.function(
    inputs=[x, y],
    outputs=cost,
    updates=[(l1.W, l1.W - learning_rate * gW1),
             (l1.b, l1.b - learning_rate * gb1),
             (l2.W, l2.W - learning_rate * gW2),
             (l2.b, l2.b - learning_rate * gb2)])
```

然后我们定义一个预测函数来输出我们最终的结果predict.

```py
# prediction
predict = theano.function(inputs=[x], outputs=l2.outputs)
```

## 4.训练

```py
for i in range(1000):
    # training
    err = train(x_data, y_data)
    if i % 50 == 0:
        print(err)
```