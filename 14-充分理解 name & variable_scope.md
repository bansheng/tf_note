[参考：TensorFlow共享变量](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variable_scope.html)
## 1. name 和 variable_scope的作用
1. name_scope:                         为了更好地管理变量的命名空间而提出的。比如在 tensorboard 中，因为引入了 name_scope， 我们的 Graph 看起来才井然有序。
2. variable_scope: 大大大部分情况下，跟 tf.get_variable() 配合使用，实现变量共享的功能。

## 2. 三种方式创建变量： tf.placeholder, tf.Variable, tf.get_variable
+ tf.placeholder() 占位符。*trainable==False*
+ tf.Variable() 一般变量用这种方式定义。*可以选择 trainable 类型*
+ tf.get_variable() 一般都是和 tf.variable_scope() 配合使用，从而实现变量共享的功能。 *可以选择 trainable 类型*
+ 这三种方式所定义的变量具有相同的类型。而且只有 tf.get_variable() 创建的变量之间会发生命名冲突


## 3. 探索 name_scope 和 variable_scope
### 1. name_scope对get_variable无影响
```
import tensorflow as tf
with tf.name_scope('outer1'):
    v1 = tf.Variable([1], name='v1')
    with tf.variable_scope('inner2'):
        v2 = tf.Variable([1], name='v2')
        v3 = tf.get_variable(name='v3', shape=[])
print('v1.name: ', v1.name)
print('v2.name: ', v2.name)
print('v3.name: ', v3.name)

with tf.name_scope('outer2'):
    v4 = tf.get_variable(name='v4',shape=[])
print('v4.name: ', v4.name)
```
```
>>>v1.name:  outer1/v1:0
>>>v2.name:  outer1/inner2/v2:0
>>>v3.name:  inner2/v3:0
>>>v4.name:  v4:0
```


### 2. get_variable和Variable的区别
```
import tensorflow as tf
sess = tf.Session()

# 拿官方的例子改动一下
def my_image_filter():
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv2_weights")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    return None

# First call creates one set of 4 variables.
result1 = my_image_filter()
# Another set of 4 variables is created in the second call.
result2 = my_image_filter()
# 获取所有的可训练变量
vs = tf.trainable_variables()
print('There are %d train_able_variables in the Graph: ' % len(vs))
for v in vs:
    print(v)
```
```
# There are 8 train_able_variables in the Graph:
<tf.Variable 'conv1_weights:0' shape=(5, 5, 32, 32) dtype=float32_ref>
<tf.Variable 'conv1_biases:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'conv2_weights:0' shape=(5, 5, 32, 32) dtype=float32_ref>
<tf.Variable 'conv2_biases:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'conv1_weights_1:0' shape=(5, 5, 32, 32) dtype=float32_ref>
<tf.Variable 'conv1_biases_1:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'conv2_weights_1:0' shape=(5, 5, 32, 32) dtype=float32_ref>
<tf.Variable 'conv2_biases_1:0' shape=(32,) dtype=float32_ref>

```
> 通过上面可以看到如果使用普通的Variable，会创建两次变量。
```
import tensorflow as tf
sess = tf.Session()

# 下面是定义一个卷积层的通用方式
def conv_relu(kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    return None


def my_image_filter():
    # 按照下面的方式定义卷积层，非常直观，而且富有层次感
    with tf.variable_scope("conv1"): 
    # 创建两个变量
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu([5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
    # 创建两个变量
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu( [5, 5, 32, 32], [32])



# 下面我们两次调用 my_image_filter 函数，但是由于引入了 变量共享机制
# 可以看到我们只是创建了一遍网络结构。
result1 = my_image_filter()
tf.get_variable_scope().reuse_variables() ##必须通过这行指定变量共享，否则抛出异常
result2 = my_image_filter()


# 看看下面，完美地实现了变量共享！！！
vs = tf.trainable_variables()
print('There are %d train_able_variables in the Graph: ' % len(vs))
for v in vs:
    print(v)
```
```
>>>There are 4 train_able_variables in the Graph: 
    <tf.Variable 'conv1/weights:0' shape=(5, 5, 32, 32) dtype=float32_ref>
    <tf.Variable 'conv1/biases:0' shape=(32,) dtype=float32_ref>
    <tf.Variable 'conv2/weights:0' shape=(5, 5, 32, 32) dtype=float32_ref>
    <tf.Variable 'conv2/biases:0' shape=(32,) dtype=float32_ref>
```
> 通过tf.variable_scope和tf.get_variable()方法，能够共享变量，而不是创建新的变量