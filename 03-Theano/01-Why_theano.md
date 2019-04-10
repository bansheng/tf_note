# 为什么我们用theano？

[TOC]

tensowflow 目前只能在 MacOS 和 Linux， theano 不仅可以在前两个系统中运行, 还可以在 Windows 系统下运行;

+ theano 可以使用 GPU 进行运算，用GPU运行比CPU快100倍左右，theano 是比较优秀的 python 模块。
+ 对于初学者来说，如果可以在 theano 和 tensorflow 中选择, 个人推荐使用 tensowflow. tensowflow 是比较商业化的模块，用起来没有theano 学术化。如果是使用机器学习进行学术性研究，网上已经有很多使用 theano 的学术性资料。所以 theano 在这种情况下是值得推荐的。

## 1. Theano的基本用法

在 theano 中学会定义矩阵 matrix 和功能 function 是一个比较重要的事, 我们在这里简单的提及了一下在 theano 将要运用到的东西.

theano 和 tensorflow 类似，都是基于建立神经网络每个组件，在组件联系起来，数据放入组件，得到结果。

```py
# 首先, 我们这次需要加载 theano 和 numpy 两个模块, 并且使用 theano 来创建 function.
import numpy as np
import theano.tensor as T
from theano import function

# 定义X和Y两个常量 (scalar)，把结构建立好之后，把结构放在function，在把数据放在function。
# basic
x = T.dscalar('x')  # 建立 x 的容器
y = T.dscalar('y')  # 建立 y 的容器
z = x+y     #  建立方程

# 使用 function 定义 theano 的方程,
# 将输入值 x, y 放在 [] 里,  输出值 z 放在后面
f = function([x, y], z)  

print(f(2,3))  # 将确切的 x, y 值放入方程中
>>> 5.0

# 使用 theano 中 的 pp (pretty-print) 能够打印出原始方程:
from theano import pp
print(pp(z))
>>>(x+y)

x = T.dmatrix('x')  # 矩阵 x 的容器
y = T.dmatrix('y')  # 矩阵 y 的容器
z = x + y   # 定义矩阵加法
f = function([x, y], z) # 定义方程

print(f(np.arange(12).reshape((3,4)), 10*np.ones((3,4))))
>>>[[10. 11. 12. 13.]
 [14. 15. 16. 17.]
 [18. 19. 20. 21.]]
```