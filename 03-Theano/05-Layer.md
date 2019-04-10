# 定义layer类

在学习了如何利用Theano定义神经网络的激活函数后，我们用一个class类来规划神经网络层的信息，这样会比较方便的我们之后的编程应用.

## 1. 要点

```py
import theano
import theano.tensor as T
import numpy as np
```

接下来我们来设计神经网络的`Layer`类。 对于神经网络的每个`Layer`，我们首先需要明确几个基本特征。 我们可以把每一层神经网络想象成一个函数，它具有输入是数据来源`input`， 输入神经元维度`in_size`，输出神经元纬度`out_size`,和指定的神经元激活函数`activation_function`。

```py
# 下面两行代码就表示了具有两层神经元的神经网络
# to define the layer like this:
l1 = Layer(inputs, 1, 10, T.nnet.relu)
l2 = Layer(l1.outputs, 10, 1, None)
```

其中，第一层网络我们命名为l1, 输入变量为inputs, 输入为1维，输出为10维，也就是说l, 含有10个神经元或节点； 激活函数为`theano.tensor.nnet.relu`函数, 当然我们也可以针对不同的问题选用别的函数, 例如`theano.tensor.nnet.nnet.sigmoid`, 详情请见Theano官方文档。

第二层网络的输入是第一层网络的输出`l1.outputs`, 所以输入的维度是10，输出是1维，激活函数我们采用one,也就是说我们采用默认的线形激活函数， l2层含有1个神经元，也就是网络的输出神经元。

以上的代码，描述并构建了一个1-10-1的神经网络（inputs-l1-l2）。

## 2. 定义层结构

接下来我们具体实现类的结构

```py
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

## 3. 细节说明

这段代码中，我们最关心的就是这个类的构造函数

```py
def __init__(self, inputs, in_size, out_size, activation_function=None)
```

和之前的例子一致，我们采用了相同的输入变量名。

接着，我们定义了W,b来代表该神经网络层的输入权值和偏置值，我们把W初始化为 由符合均值为0， 方差为1高斯分布的随机变量值组成的`in_size-by-out_size`的矩阵; b初始化为值为0.1的`out_put-by-1`的向量。 (当然，我们也可以采用不同的初始化方法，这里我们暂时不讨论初始化权值对最终神经网络训练的影响)。

```py
self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))
self.b = theano.shared(np.zeros((out_size, )) + 0.1)
```

首先我们要计算所有神经元的输入矩阵, 也就是输入inputs与输入权值W的点乘（dot product）在加上偏置值b：

```py
self.Wx_plus_b = T.dot(inputs, self.W) + self.b
```

然后，我们需要根据我们构造神经层指定的激活函数类型`activation_function`,来计算神经层的输出向量。 这里我们假设如果`activation_function`是`None`， 那就是该层神经元采用线形输出；如果是其他Theano的激活函数，就把`Wx_plus_b`作为该层激活函数的输入，同时函数的输出即为神经层的输出：

```py
self.activation_function = activation_function
if activation_function is None:
    self.outputs = self.Wx_plus_b
else:
    self.outputs = self.activation_function(self.Wx_plus_b)
```