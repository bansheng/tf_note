## 1. 人工神经网络
### 1.1 神经元
每个神经元接受线性组合的输入后，最开始只是简单的线性加权，后来给每个神经元加上了非线性的激活函数，从而进行非线性变换后输出。  
每两个神经元之间的连接代表加权值，称之为权重（weight）。不同的权重和激活函数，则会导致神经网络不同的输出。
![neuron](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQyvr2leOsUoInOk95OXchRPHrYptmI3gYE-lJN_sEQIjrQhnR-)
```math
f(x) = \sum_{i=1}^{n}{w_i*x_i} + b
再通过激活函数
最后产生输出
```

### 1.2 激活函数
常见的包括sigmod, tanh, relu。
sigmod/tanh比较常见用于全连接层，而relu多用于卷积层（卷积层默认激活函数）

## 2. 神经网络NN
![image](https://cdn-images-1.medium.com/max/1200/1*CcQPggEbLgej32mVF2lalg.png)

包括输入层、输出层以及多个隐藏层
![image](https://cdn-images-1.medium.com/max/1600/1*7QYEGFnWvharpRe5-k3T6A.jpeg)
每一层都可能由单个或多个神经元组成，每一层的输出将会作为下一层的输入数据。比如下图中间隐藏层来说，隐藏层的3个神经元a1、a2、a3皆各自接受来自多个不同权重的输入（因为有x1、x2、x3这三个输入，所以a1 a2 a3都会接受x1 x2 x3各自分别赋予的权重，即几个输入则几个权重），接着，a1、a2、a3又在自身各自不同权重的影响下 成为的输出层的输入，最终由输出层输出最终结果。

此外，输入层和隐藏层都存在一个偏置（bias unit)，所以上图中也增加了偏置项：x0、a0

层和层之间是全连接的结构，同一层的神经元之间没有连接。

## 3.卷积神经网络之层级结构
1. 输入层
2. CONV 卷积计算层 RELU激励层 POOL池化层 
3. FC全连接层


## 4. CNN之卷积计算层
### 4.1 什么是卷积
对图像（不同的数据窗口数据）和滤波矩阵（一组固定的权重：因为每个神经元的多个权重固定，所以又可以看做一个恒定的滤波器filter）做内积（逐个元素相乘再求和）的操作就是所谓的『卷积』操作，也是卷积神经网络的名字来源。

### 4.2 卷积的计算
![image](http://x-wei.github.io/images/Ng_DLMooc_c4wk1/pasted_image001.png)

## 5. CNN之激励层和池化层
### 5.1 RELU激励层
实际梯度下降中，sigmoid容易饱和、造成终止梯度传递，且没有0中心化。ReLU的优点是收敛快，求梯度简单。

### 5.2 池化pool层
池化，简言之，即取区域平均或最大
![image](http://cs231n.github.io/assets/cnn/maxpool.jpeg)

## 6. tf.contrib.framework.arg_scope
```
tf.contrib.framework.arg_scope(
    list_ops_or_scope,
    **kwargs
)
# Stores the default arguments for the given set of list_ops.
# 为输入的op设置存储默认参数值
----------------------------
 with tf.contrib.framework.arg_scope(
    [tf.contrib.layers.conv2d],
    padding='SAME',
    weights_regularizer=slim.l2_regularizer(weight_decay),
    weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                        uniform=False, seed=None,
                                                                        dtype=tf.float32),
    activation_fn=tf.nn.relu) as sc:
return sc
```