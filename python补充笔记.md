# python知识补充笔记

Tags: python 语言

[TOC]

## 1. @cleanup_handle

> 装饰器，指示程序在结束之前释放数据占据的内存。

## 2. python3 装饰器

### 2.1 代码复用

> 我们在程序执行的过程中如果想要实现一个类似日志的功能

```python
def foo():
    print('I am foo')

# 加入日志
def foo():
    logging.info('foo is running')
    print('I am foo')
```

> 如果在fun1() fun2()都想实现类似的功能呢？能不能实现代码的复用，而不是重复的在每个地方都加上类似的功能。

```python
def use_logging(func):
    logging.warn("%s is running" % func.__name__)
    func()

def foo():
    print('i am foo')

use_logging(foo)
```

> 这样做逻辑上是没问题的，功能是实现了，但是我们调用的时候不再是调用真正的业务逻辑 foo 函数，而是换成了 use_logging 函数，这就破坏了原有的代码结构， 现在我们不得不每次都要把原来的那个 foo 函数作为参数传递给 use_logging 函数，那么有没有更好的方式的呢？当然有，答案就是装饰器。

### 2.2 简单装饰器

```python
def use_logging(func):

    def wrapper():
        logging.warn("%s is running" % func.__name__)
        return func()   # 把 foo 当做参数传递进来时，执行func()就相当于执行foo()
    return wrapper

def foo():
    print('i am foo')

foo = use_logging(foo)  # 因为装饰器 use_logging(foo) 返回的时函数对象 wrapper，这条语句相当于  foo = wrapper
foo()                   # 执行foo()就相当于执行 wrapper()
```

> use_logging 就是一个装饰器，它一个普通的函数，它把执行真正业务逻辑的函数 func 包裹在其中，看起来像 foo 被 use_logging 装饰了一样，use_logging 返回的也是一个函数，这个函数的名字叫 wrapper。在这个例子中，函数进入和退出时 ，被称为一个横切面，这种编程方式被称为面向切面的编程。

### 2.2语法糖

```python
def use_logging(func):

    def wrapper():
        logging.warn("%s is running" % func.__name__)
        return func()
    return wrapper

@use_logging
def foo():
    print("i am foo")

foo()
```

>如上所示，有了 @ ，我们就可以省去foo = use_logging(foo)这一句了，直接调用 foo() 即可得到想要的结果。你们看到了没有，foo() 函数不需要做任何修改，只需在定义的地方加上装饰器，调用的时候还是和以前一样，如果我们有其他的类似函数，我们可以继续调用装饰器来修饰函数，而不用重复修改函数或者增加新的封装。这样，我们就提高了程序的可重复利用性，并增加了程序的可读性。
>
>装饰器在 Python 使用如此方便都要归因于 Python 的函数能像普通的对象一样能作为参数传递给其他函数，可以被赋值给其他变量，可以作为返回值，可以被定义在另外一个函数内。

### 2.3 装饰器顺序

> 可以同时定义多个装饰器

```python
@a
@b
@c
def f ():
    pass
```

> 它的执行顺序是从里到外，最先调用最里层的装饰器，最后调用最外层的装饰器，它等效于

```python
f = a(b(c(f)))
```

## 3. try-except-raise

raise用于抛出异常。如果raise后面没有对象，代表把捕获的对象重新抛出。

```shell
>>> try:
...     raise NameError('HiThere')
... except NameError:
...     print('An exception flew by!')
...     raise
...
An exception flew by!
Traceback (most recent call last):
  File "<stdin>", line 2, in ?
NameError: HiThere
```

## 4. argparse.ArgumentParser

python argparse用法总结

### 4.1 argparse介绍

是python的一个命令行解析包，非常编写可读性非常好的程序

### 4.2 基本用法

prog.py是我在linux下测试argparse的文件，放在/tmp目录下，其内容如下：

```python
#!/usr/bin/env python
# encoding: utf-8

import argparse
parser = argparse.ArgumentParser()
parser.parse_args()
```

测试：

```shell
yarving@yarving-VirtualBox /tmp $ python prog.py

yarving@yarving-VirtualBox /tmp $ python prog.py --help
usage: prog.py [-h]

optional arguments:
  -h, --help  show this help message and exit

yarving@yarving-VirtualBox /tmp $ python prog.py -v
usage: prog.py [-h]
prog.py: error: unrecognized arguments: -v

yarving@yarving-VirtualBox /tmp $ python prog.py foo
usage: prog.py [-h]
prog.py: error: unrecognized arguments: foo
```

+ 第一个没有任何输出和出错
+ 第二个测试为打印帮助信息，argparse会自动生成帮助文档
+ 第三个测试为未定义的-v参数，会出错
+ 第四个测试为未定义的参数foo，出错

### 4.3 positional arguments

positional arguments为英文定义，中文名叫有翻译为定位参数的，用法是不用带`-`就可用。
修改prog.py的内容如下：

```python
#!/usr/bin/env python
# encoding: utf-8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("echo")
args = parser.parse_args()
print(args.echo)
```

执行测试如下

```shell
yarving@yarving-VirtualBox /tmp $ python prog.py
usage: prog.py [-h] echo
prog.py: error: too few arguments

yarving@yarving-VirtualBox /tmp $ python prog.py -h
usage: prog.py [-h] echo

positional arguments:
  echo

optional arguments:
  -h, --help  show this help message and exit

yarving@yarving-VirtualBox /tmp $ python prog.py hahahaha
hahahaha
```

定义了一个叫echo的参数，默认必选

+ 第一个测试为不带参数，由于echo参数为空，所以报错，并给出用法（usage）和错误信息
+ 第二个测试为打印帮助信息
+ 第三个测试为正常用法，回显了输入字符串hahahaha

### 4.4 optional arguments

中文名叫可选参数，有两种方式：

一种是通过一个-来指定的短参数，如-h；
一种是通过--来指定的长参数，如--help
这两种方式可以同存，也可以只存在一个，修改prog.py内容如下：

```python
#!/usr/bin/env python
# encoding: utf-8
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbosity", help="increase output verbosity")
args = parser.parse_args()
if args.verbosity:
        print(verbosity turned on)
```

>注意这一行：parser.add_argument("-v", "--verbosity", help="increase output verbosity")

定义了可选参数-v或--verbosity，通过解析后，其值保存在args.verbosity变量中
用法如下：

```python
yarving@yarving-VirtualBox /tmp $ python prog.py -v 1
verbosity turned on

yarving@yarving-VirtualBox /tmp $ python prog.py --verbosity 1
verbosity turned on

yarving@yarving-VirtualBox /tmp $ python prog.py -h
usage: prog.py [-h] [-v VERBOSITY]

optional arguments:
  -h, --help            show this help message and exit
  -v VERBOSITY, --verbosity VERBOSITY
                        increase output verbosity

yarving@yarving-VirtualBox /tmp $ python prog.py -v
usage: prog.py [-h] [-v VERBOSITY]
prog.py: error: argument -v/--verbosity: expected one argument
```

+ 测试1中，通过-v来指定参数值
+ 测试2中，通过--verbosity来指定参数值
+ 测试3中，通过-h来打印帮助信息
+ 测试4中，没有给-v指定参数值，所以会报错

### 4.5 action='store_true'

上一个用法中-v必须指定参数值，否则就会报错，有没有像-h那样，不需要指定参数值的呢，答案是有，通过定义参数时指定action="store_true"即可，用法如下

```python
#!/usr/bin/env python
# encoding: utf-8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()
if args.verbose:
        print("verbosity turned on")
```

测试：

```shell
yarving@yarving-VirtualBox /tmp $ python prog.py -v
verbosity turned on

yarving@yarving-VirtualBox /tmp $ python prog.py -h
usage: prog.py [-h] [-v]

optional arguments:
  -h, --help     show this help message and exit
  -v, --verbose  increase output verbosity
```

第一个例子中，-v没有指定任何参数也可，**其实存的是True和False**，如果出现，则其值为True，否则为False

### 4.6 类型 type

默认的参数类型为str，如果要进行数学计算，需要对参数进行解析后进行类型转换，如果不能转换则需要报错，这样比较麻烦
argparse提供了对参数类型的解析，如果类型不符合，则直接报错。如下是对参数进行平方计算的程序：

```python
#!/usr/bin/env python
# encoding: utf-8
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('x', type=int, help="the base")
args = parser.parse_args()
answer = args.x ** 2
print(answer)
```

测试

```python
yarving@yarving-VirtualBox /tmp $ python prog.py 2
4

yarving@yarving-VirtualBox /tmp $ python prog.py two
usage: prog.py [-h] x
prog.py: error: argument x: invalid int value: 'two'

yarving@yarving-VirtualBox /tmp $ python prog.py -h
usage: prog.py [-h] x

positional arguments:
  x           the base

optional arguments:
  -h, --help  show this help message and exit
```

+ 第一个测试为计算2的平方数，类型为int，正常
+ 第二个测试为一个非int数，报错
+ 第三个为打印帮助信息

### 4.7 可选值choices=[]

5中的action的例子中定义了默认值为True和False的方式，如果要限定某个值的取值范围，比如6中的整形，限定其取值范围为0， 1， 2，该如何进行呢？
修改prog.py文件如下：

```python
#!/usr/bin/env python
# encoding: utf-8
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("square", type=int,
                    help="display a square of a given number")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
                    help="increase output verbosity")
args = parser.parse_args()
answer = args.square**2
if args.verbosity == 2:
    print("the square of {} equals {}".format(args.square, answer))
elif args.verbosity == 1:
    print("{}^2 == {}".format(args.square, answer))
else:
    print(answer)
```

测试如下：

```shell
yarving@yarving-VirtualBox /tmp $ python prog.py 4 -v 0
16
yarving@yarving-VirtualBox /tmp $ python prog.py 4 -v 1
4^2 == 16
yarving@yarving-VirtualBox /tmp $ python prog.py 4 -v 2
the square of 4 equals 16
yarving@yarving-VirtualBox /tmp $ python prog.py 4 -v 3
usage: prog.py [-h] [-v {0,1,2}] square
prog.py: error: argument -v/--verbosity: invalid choice: 3 (choose from 0, 1, 2)
yarving@yarving-VirtualBox /tmp $ python prog.py -h
usage: prog.py [-h] [-v {0,1,2}] square

positional arguments:
  square                display a square of a given number

optional arguments:
  -h, --help            show this help message and exit
  -v {0,1,2}, --verbosity {0,1,2}
                        increase output verbosity
```

+ 测试1， 2， 3 为可选值范围，通过其值，打印不同的格式输出；
+ 测试4的verbosity值不在可选值范围内，打印错误
+ 测试5打印帮助信息

### 4.8 自定义帮助信息help

上面很多例子中都为help赋值，如

```python
parser.add_argument("square", type=int, help="display a square of a given number")
```

在打印输出时，会有如下内容

```shell
positional arguments:
  square           display a square of a given number
```

也就是help为什么，打印输出时，就会显示什么

### 4.9 程序用法帮助

8中介绍了为每个参数定义帮助文档，那么给整个程序定义帮助文档该怎么进行呢？
通过argparse.ArgumentParser(description="calculate X to the power of Y")即可
修改prog.py内容如下：

```python
#!/usr/bin/env python
# encoding: utf-8
import argparse

parser = argparse.ArgumentParser(description="calculate X to the power of Y")
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("x", type=int, help="the base")
parser.add_argument("y", type=int, help="the exponent")
args = parser.parse_args()
answer = args.x**args.y

if args.quiet:
    print(answer)
elif args.verbose:
    print("{} to the power {} equals {}".format(args.x, args.y, answer))
else:
    print("{}^{} == {}".format(args.x, args.y, answer))
```

打印帮助信息时即显示calculate X to the power of Y

```python
yarving@yarving-VirtualBox /tmp $ python prog.py -h
usage: prog.py [-h] [-v | -q] x y

calculate X to the power of Y

positional arguments:
  x              the base
  y              the exponent

optional arguments:
  -h, --help     show this help message and exit
  -v, --verbose
  -q, --quiet
```

### 4.10 互斥参数

在上个例子中介绍了互斥的参数

```python
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet", action="store_true")
```

第一行定义了一个互斥组，第二、三行在互斥组中添加了-v和-q两个参数，用上个例子中的程序进行如下测试：

```shell
yarving@yarving-VirtualBox /tmp $ python prog.py 4 2
4^2 == 16
yarving@yarving-VirtualBox /tmp $ python prog.py 4 2 -v
4 to the power 2 equals 16
yarving@yarving-VirtualBox /tmp $ python prog.py 4 2 -q
16
yarving@yarving-VirtualBox /tmp $ python prog.py 4 2 -q -v
```

可以看出，-q和-v不出现，或仅出现一个都可以，同时出现就会报错。
可定义多个互斥组

### 4.11参数默认值

介绍了这么多，有没有参数默认值该如何定义呢？
修改prog.py内容如下：

```python
#!/usr/bin/env python
# encoding: utf-8
import argparse

parser = argparse.ArgumentParser(description="calculate X to the power of Y")
parser.add_argument("square", type=int,
                    help="display a square of a given number")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=1,
                    help="increase output verbosity")
args = parser.parse_args()
answer = args.square**2
if args.verbosity == 2:
    print("the square of {} equals {}".format(args.square, answer))
elif args.verbosity == 1:
    print("{}^2 == {}".format(args.square, answer))
else:
    print(answer)
```

测试结果如下

```python
yarving@yarving-VirtualBox /tmp $ python prog.py 8
8^2 == 64
yarving@yarving-VirtualBox /tmp $ python prog.py 8 -v 0
64
yarving@yarving-VirtualBox /tmp $ python prog.py 8 -v 1
8^2 == 64
yarving@yarving-VirtualBox /tmp $ python prog.py 8 -v 2
the square of 8 equals 64
```

可以看到如果不指定-v的值，args.verbosity的值默认为1，为了更清楚的看到默认值，也可以直接打印进行测试。

## 5. zip() enumetate() 和字典在for in循环中的区别

### 5.1 字典

```python
c = {1:'a', 2:'b', 3:'c'}
for key in c.keys():
    print(key)
for value in c.values():
    print(value)
for key in c:
    print(key)
for key in c.items():
    print(key)
for key, value in c.items():
    print(key, value)

>>1
2
3
a
b
c
1
2
3
(1, 'a')
(2, 'b')
(3, 'c')
1 a
2 b
3 c
```

> 总结：总共5种访问形式,直接访问，访问的是键值，可以单独访问键值，还能访问item，type为元祖

### 5.2 enumerate()

```python
a = [1, 2, 3]
for value in enumerate(a):
    print(value)
for index, value in enumerate(a):
    print(index, value)
>>>(0, 1)
(1, 2)
(2, 3)
0 1
1 2
2 3
```

enumerate()函数将可迭代的数据包装，加上编号，单独访问的访问的是元祖。

### 5.3 zip()

```python
a = [1, 2, 3, 4, 5, 6]
b = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
for value in zip(a, b):
    print(value)

b = ['a', 'b', 'c', 'd', 'e', 'f']
for value in zip(a, b):
    print(value)

b = ['a', 'b', 'c', 'd', 'e']
for value in zip(a, b):
    print(value)

>>>(1, 'a')
(2, 'b')
(3, 'c')
(4, 'd')
(5, 'e')
(6, 'f')

(1, 'a')
(2, 'b')
(3, 'c')
(4, 'd')
(5, 'e')
(6, 'f')

(1, 'a')
(2, 'b')
(3, 'c')
(4, 'd')
(5, 'e')
```

zip将两个可迭代的数据形式结合，按顺序组成字典，以迭代次数较少的为上限，单独访问为元祖
