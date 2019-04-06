## 1. 线性回归linear regression
线性回归用来度量变量y和至少一个变量x之间的关系，线性回归的使用的简单性在于，用于解释新模型和将数据映射到另一个空间。


## 2. tf.squared_difference
返回差的平方
```
tf.math.squared_difference(
    x,
    y,
    name=None
)
```
返回`(x-y)(x-y)`


## 3.tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
梯度下降
```
minimize(
    loss,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name=None,
    grad_loss=None
)
```
通过更新var_list添加操作来最小化loss

## 4. tf.placeholder
```
tf.placeholder(
    dtype,
    shape=None,
    name=None
)
```
+ 为一个总被feed的Tensor插入一个占位符
+ 返回值：一个用于feed数据的Tensor，不能直接计算，需要在Session.run()的时候feed数据。