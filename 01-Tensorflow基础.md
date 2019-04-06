## 1. 日志文件存储
为了使用`tensorboard`可视化工具，我们需要创建绝对文件夹路径，存储event文件。通过`FLAG`变量访问。
```
# The default path for saving event files is the same folder of this python file.
tf.app.flags.DEFINE_string(
'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
'Directory where event logs are written to.')

# Store all elements in FLAG structure!
FLAGS = tf.app.flags.FLAGS
```

## 2. tf.app.flags
+ `tf.app.flags`模块是Tensorflow提供的功能，用于为Tensorflow程序实现命令行标志  
+ 使用`tf.app.run()`的时候，使用`tf.app.flags`能后在线程间方便的调动变量
+ 为参数设置默认，用于代替`argparse or sys.argv.`
+ **可以在命令行给flags里面的变量赋值**


## 3. 定义常量
```
# Defining some sentence!
welcome = tf.constant('Welcome to TensorFlow world!')

# Defining some constant values
a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name="b")

# Some basic operations
x = tf.add(a, b, name="add")
y = tf.div(a, b, name="divide")
```
这样定义出的常量是一个`Tensor`，`name=somename`这个属性便于在tensorboard里面可视化。

## 4. session
`tf.Session()`是用来进行计算的环境。
```
# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print("output: ", sess.run(welcome))

# Closing the writer.
writer.close()
sess.close()
```
`sess.run()`里面必须是一个Tensor的计算，否则无法执行。

## 5. tensorboard
`tensorborad --logdir='absPath/to/logdir'`