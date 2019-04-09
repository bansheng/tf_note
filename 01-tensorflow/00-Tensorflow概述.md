# Tensorflow概述

## 1. 图与会话

TensorFlow 使用数据流图将计算表示为独立的指令之间的依赖关系。这可生成低级别的编程模型，在该模型中，您首先定义数据流图，然后创建 TensorFlow 会话，以便在一组本地和远程设备上运行图的各个部分。

## 2. 为什么使用数据流图

![数据流图](https://tensorflow.google.cn/images/tensors_flowing.gif)

数据流是一种用于并行计算的常用编程模型。在数据流图中，**节点表示计算单元**，**边表示计算使用或产生的数据**。例如，在 TensorFlow 图中，tf.matmul 操作对应于单个节点，该节点具有两个传入边（要相乘的矩阵）和一个传出边（乘法结果）。

优势

1. 并行处理
2. 分布式执行
3. 编译
4. 可移植性

## 2. 什么是`tf.Graph`

`tf.Graph`包含两类信息

+ 图结构，图的节点和边缘，表示各个操作组合在一起的方式，但不规定它们的使用方式。
+ 图集合。`TensorFlow`提供了一种在`tf.Graph` 中存储元数据集合的通用机制。`tf.add_to_collection`函数允许您将对象列表与一个键关联（其中 `tf.GraphKeys`定义了部分标准键），`tf.get_collection`允许您查询与某个键关联的所有对象.`TensorFlow`库的许多部分会使用此设施资源：例如，当您创建`tf.Variable`时，系统会默认将其添加到表示“全局变量”和“可训练变量”的集合中。当您后续创建`tf.train.Saver`或`tf.train.Optimizer`时，这些集合中的变量将用作默认参数。

## 3. 构建`tf.Graph`

+ *Tensor保存的是数据信息*
+ *Operation(op)保存的是计算单元*  

大多数`TensorFlow`程序都以数据流图构建阶段开始。在此阶段，您会调用 TensorFlow API 函数，这些函数可构建新的tf.Operation（节点）和tf.Tensor（边）对象并将它们添加到`tf.Graph`实例中。TensorFlow 提供了一个默认图，此图是同一上下文中的所有 API 函数的明确参数。例如：  

> 调用 tf.constant(42.0) 可创建单个 tf.Operation，该操作可以生成值 42.0，将该值添加到默认图中，并返回表示常量值的 tf.Tensor。
>
> 调用 tf.matmul(x, y) 可创建单个 tf.Operation，该操作会将 tf.Tensor 对象 x 和 y 的值相乘，将其添加到默认图中，并返回表示乘法运算结果的 tf.Tensor。
>
> 执行 v = tf.Variable(0) 可向图添加一个 tf.Operation，该操作可以存储一个可写入的张量值，该值在多个 tf.Session.run 调用之间保持恒定。tf.Variable 对象会封装此操作，并可以像张量一样使用，即读取已存储值的当前值。tf.Variable 对象也具有 assign 和 assign_add 等方法，这些方法可创建 tf.Operation 对象，这些对象在执行时将更新已存储的值。（请参阅变量了解关于变量的更多信息。）
>
> 调用 tf.train.Optimizer.minimize 可将操作和张量添加到计算梯度的默认图中，并返回一个 tf.Operation，该操作在运行时会将这些梯度应用到一组变量上。

注意：调用 TensorFlow API 中的大多数函数只会将操作和张量添加到默认图中，而**不会执行实际计算**。您应编写这些函数，直到拥有表示整个计算（例如执行梯度下降法的一步）的 `tf.Tensor`或`tf.Operation`，然后将该对象传递给 `tf.Session`以执行计算。更多详情请参阅“在 `tf.Session`中执行图”部分。

## 4. 命名指令

`tf.Graph`对象会定义一个命名空间（为其包含的 `tf.Operation`对象）。TensorFlow 会自动为您的图中的每个指令选择一个唯一名称，但您也可以指定描述性名称，使您的程序阅读和调试起来更轻松。TensorFlow API 提供两种方法来覆盖操作名称：

+ 如果 API 函数会创建新的`tf.Operation`或返回新的`tf.Tensor`，则会接受可选 name 参数。例如`tf.constant(42.0, name="answer")`会创建一个新的`tf.Operation`（名为 "answer"）并返回一个`tf.Tensor`（名为 "answer:0"）。如果默认图已包含名为 "answer" 的操作，则 TensorFlow 会在名称上附加 "_1"、"_2" 等字符，以便让名称具有唯一性。
+ 借助`tf.name_scope`函数，您可以向在特定上下文中创建的所有操作添加名称作用域前缀。当前名称作用域前缀是一个用 "/" 分隔的名称列表，其中包含所有活跃`tf.name_scope` 上下文管理器的名称。如果某个名称作用域已在当前上下文中被占用，TensorFlow 将在该作用域上附加 "_1"、"_2" 等字符。

```py
c_0 = tf.constant(0, name="c")  # => operation named "c"

# Already-used names will be "uniquified".
c_1 = tf.constant(2, name="c")  # => operation named "c_1"

# Name scopes add a prefix to all operations created in the same context.
with tf.name_scope("outer"):
  c_2 = tf.constant(2, name="c")  # => operation named "outer/c"

  # Name scopes nest like paths in a hierarchical file system.
  with tf.name_scope("inner"):
    c_3 = tf.constant(3, name="c")  # => operation named "outer/inner/c"

  # Exiting a name scope context will return to the previous prefix.
  c_4 = tf.constant(4, name="c")  # => operation named "outer/c_1"

  # Already-used name scopes will be "uniquified".
  with tf.name_scope("inner"):
    c_5 = tf.constant(5, name="c")  # => operation named "outer/inner_1/c"
```

图可视化工具使用名称范围来为指令分组并降低图的视觉复杂性。  
请注意`tf.Tensor`对象以输出张量的`tf.Operation`明确命名。张量名称的形式为 "<OP_NAME>:< i >"，其中：

+ "<OP_NAME>" 是生成该张量的操作的名称。
+ "< i >" 是一个整数，表示该张量在操作的输出中的索引。

## 5. 类似于张量的对象

许多 TensorFlow 操作都会接受一个或多个`tf.Tensor`对象作为参数。例如,`tf.matmul`接受两个`tf.Tensor` 对象,`tf.add_n`接受一个具有 n 个 `tf.Tensor`对象的列表。为了方便起见，这些函数将接受类张量对象来取代`tf.Tensor`，并将它明确转换为`tf.Tensor`通过 tf.convert_to_tensor 方法）。类张量对象包括以下类型的元素：

1. tf.Tensor
2. tf.Variable
3. numpy.ndarray
4. list（以及类似于张量的对象的列表）
5. 标量 Python 类型：bool、float、int、str

## [6. 将操作放到不同设备上面](https://tensorflow.google.cn/guide/graphs#placing_operations_on_different_devices)

## 7. 创建tf.Session()

```python
# Create a default in-process session.
with tf.Session() as sess:
  # ...

# Create a remote session.
with tf.Session("grpc://example.org:2222"):
  # ...
```

with语句隐式的close了Session()

## 8. 使用`tf.Session.run`执行操作

`tf.Session.run`方法是运行`tf.Operation`或评估 `tf.Tensor`的主要机制。您可以将一个或多个 `tf.Operation`或`tf.Tensor`对象传递到 `tf.Session.run`，TensorFlow 将执行计算结果所需的操作。

`tf.Session.run`要求您指定一组 fetch，这些 fetch 可确定返回值，并且可能是`tf.Operation、tf.Tensor` 或类张量类型，例如`tf.Variable`。这些 fetch 决定了必须执行哪些子图（属于整体 tf.Graph）以生成结果：该子图包含 fetch 列表中指定的所有操作，以及其输出用于计算fetch 值的所有操作。例如，以下代码段说明了 `tf.Session.run`的不同参数如何导致执行不同的子图：

```py
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer

with tf.Session() as sess:
  # Run the initializer on `w`.
  sess.run(init_op)

  # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
  # the result of the computation.
  print(sess.run(output))

  # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
  # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
  # op. Both `y_val` and `output_val` will be NumPy arrays.
  y_val, output_val = sess.run([y, output])
```

`tf.Session.run`也可以选择接受`feed_dict`参数，该字典是从`tf.Tensor`对象（通常是 `tf.placeholder`张量）到在执行时会替换这些张量的值（通常是 Python 标量、列表或 NumPy 数组）的映射（{X: data}。例如：

```py
# Define a placeholder that expects a vector of three floating-point values,
# and a computation that depends on it.
x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
  # Feeding a value changes the result that is returned when you evaluate `y`.
  print(sess.run(y, {x: [1.0, 2.0, 3.0]}))  # => "[1.0, 4.0, 9.0]"
  print(sess.run(y, {x: [0.0, 0.0, 5.0]}))  # => "[0.0, 0.0, 25.0]"

  # Raises <a href="../api_docs/python/tf/errors/InvalidArgumentError"><code>tf.errors.InvalidArgumentError</code></a>, because you must feed a value for
  # a `tf.placeholder()` when evaluating a tensor that depends on it.
  sess.run(y)

  # Raises `ValueError`, because the shape of `37.0` does not match the shape
  # of placeholder `x`.
  sess.run(y, {x: 37.0})
```