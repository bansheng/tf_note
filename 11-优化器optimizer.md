## 1.tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
梯度下降优化器
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

## 2. tf.train.exponential_decay
**指数衰减法**   
为解决设定学习率(learning rate)问题，提供了指数衰减法来解决。

通过tf.train.exponential_decay函数实现指数衰减学习率。

    1. 首先使用较大学习率(目的：为快速得到一个比较优的解);
    2. 然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定);
```
tf.train.exponential_decay(
    learning_rate,
    global_step,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None
)
返回的是衰减后的学习率
decayed_learning_rate 
= learning_rate * decay_rate ^ (global_step / decay_steps)

如果staircase为True，则(global_step / decay_steps)为整数时才衰减，这样衰减的函数变为分段函数
```
```
Example: decay every 100000 steps with a base of 0.96:

...
global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
)
```
