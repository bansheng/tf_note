### 1. 逻辑回归
与线性回归不同，逻辑回归简单而言就是分类问题。

### 2. tf.one_hot()
```
tf.one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)
```
升维操作，将n维变量升为n+1维。
`indice`的大小为a1 * a2 * a3 * ...
+ axis=-1 输出为a1 * a2 * a3 * .. * depth
+ axis=0 输出为depth * a1 * a2 * a3 * ..
+ axis=1 输出为a1 * a2 * a3 * depth * ..
+ ..  

on_value默认为1
off_value默认为0

输出的第depth维第indice位(indice < depth && indice > 0, 否则全部输出off_value)值为on_value，其余位置为off_value。

```
indices = [0, 1, 2]
depth = 3
tf.one_hot(indices, depth)  # output: [3 x 3]
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]

indices = [0, 2, -1, 1]
depth = 3
tf.one_hot(indices, depth,
           on_value=5.0, off_value=0.0,
           axis=-1)  # output: [4 x 3]
# [[5.0, 0.0, 0.0],  # one_hot(0)
#  [0.0, 0.0, 5.0],  # one_hot(2)
#  [0.0, 0.0, 0.0],  # one_hot(-1)
#  [0.0, 5.0, 0.0]]  # one_hot(1)

indices = [[0, 2], [1, -1]]
depth = 3
tf.one_hot(indices, depth,
           on_value=1.0, off_value=0.0,
           axis=-1)  # output: [2 x 2 x 3]
# [[[1.0, 0.0, 0.0],   # one_hot(0)
#   [0.0, 0.0, 1.0]],  # one_hot(2)
#  [[0.0, 1.0, 0.0],   # one_hot(1)
#   [0.0, 0.0, 0.0]]]  # one_hot(-1)
```

### 3. 逻辑回归
1. 构造预测函数
```math
    sigmod函数
    
    hθ(x)=g(z)= 1/( 1 + e ^(-z) )
```

2. 构造cost函数
```math
    p(y=1|x;θ) = hθ(x)
    
    p(y=0|x;θ) = 1 - hθ(x) 
```
由于对数损失函数（logarithmic loss function) 或对数似然损失函数(log-likehood loss function)  
```math
    L(Y,P(Y|X))=−logP(Y|X)
```
则
```math
    cost(y=1|x;θ) = -㏒(hθ(x))
    
    cost(y=0|x;θ) = -㏒(1 - hθ(x))
```
接下来就是J(θ)函数，总的损失函数
```math
    J(θ) = \frac{1}{m} \sum_{i=1}^{m}{Cost(h_θ(x^{i} ), y^{i}})
    = \frac{1}{m} \sum_{i=1}^{m}{y^{i} \log (h_θ(x^i)) + (1- y^{i}) \log (1 - h_θ(x^i))}
```

3. 梯度下降法求θ的最小值
```math
    θ_j = θ_j - α \frac{\vartheta}{\vartheta θ_j}{J(θ)}
    
     θ_j = θ_j - α \frac{\vartheta}{\vartheta θ_j}{(h_θ(x^i)) - y^{i}) x^i_j }
```