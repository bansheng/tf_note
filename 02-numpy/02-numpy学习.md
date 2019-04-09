# numpy学习

[TOC]

## 1. numpy 属性

+ ndim：维度
+ shape：行数和列数
+ size：元素个数

```py
import numpy as  np
array = np.array([[1,2,3], [4,5,6]]) #列表转换为矩阵
print(array)
print('ndim', array.ndim)
print('shape', array.shape)
print('size', array.size)

>>>[[1 2 3]
 [4 5 6]]
ndim 2
shape (2, 3)
size 6
```

## 2. numpy创建array

### 2.1 关键字

+ array：从列表创建数组
+ dtype：指定数据类型
+ zeros：创建数据全为0
+ ones：创建数据全为1
+ empty：创建数据接近0
+ arrange：按指定范围创建数据
+ linspace：创建线段

### 2.2 array从列表创建数组

```py
a = np.array([2,23,4])  # list 1d
print(a)
# [2 23 4]
```

### 2.3 dtype 指定数据类型

默认int， float数据长度为64

```py
a = np.array([2,23,4],dtype=np.int)
print(a.dtype)
>>> int64

a = np.array([2,23,4],dtype=np.int32)
print(a.dtype)
>>>int32

a = np.array([2,23,4],dtype=np.float)
print(a.dtype)
>>>float64

a = np.array([2,23,4],dtype=np.float32)
print(a.dtype)
>>>float32
```

### 2.4 特殊矩阵

```py
# 创建全0矩阵
a = np.zeros((3,4)) # 数据全为0，3行4列
>>>array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])

# 创建全1矩阵
a = np.ones((3,4), dtype=float)
>>>array([[ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.]])

# 创建全空数组 其实每个数据都是0, 类型为float64
a = np.empty((3,4))
>>>array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])

# 创建序列数组arange 参数分别为开始值，结束值，步长
# [start, end)
a = np.arange(0, 10, 2)
>>>array([0, 2, 4, 6, 8])

# 用 linspace 创建线段型数据:
# # 开始端1，结束端10，且分割成9个数据，生成线段
a = np.linspace(1, 10, 9)
>>>array([  1.   ,   2.125,   3.25 ,   4.375,   5.5  ,   6.625,   7.75 ,    8.875,  10.   ])

# 完全矩阵
c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"
>>>[[7 7]
 [7 7]]

# 特征矩阵
d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"
>>>[[ 1.  0.]
 [ 0.  1.]]
```

### 2.5 使用reshape改变矩阵的shape

reshape函数里面第一个参数为一个元祖，指定新的shape大小
原理就是先把矩阵拉直为1维矩阵，再进行升维

```py
import numpy as np
a = np.arange(0,10,1).reshape((2, 5))
>>>array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])

a = np.linspace(0, 10, 9).reshape((3, -1))
>>>array([[  0.  ,   1.25,   2.5 ],
       [  3.75,   5.  ,   6.25],
       [  7.5 ,   8.75,  10.  ]])
```

## 3. numpy基础运算1

### 3.1 矩阵元素运算

这里的所有运算针对的都是矩阵里面的单独的元素

```py
import numpy as np
a=np.array([10,20,30,40])   # array([10, 20, 30, 40])
b=np.arange(4)              # array([0, 1, 2, 3])
# 矩阵减法
c=a-b  # array([10, 19, 28, 37])
c = np.subtrac(a, b)
print(c)
>>>[ 10 19 28 37 ]

# 矩阵加法
c = a + b
c = np.add(a, b)
print(c)
>>>[ 10 21 32 43 ]

# 矩阵乘法
c = a * b
c = np.multiply(a, b)
print(c)
>>>[  0  20  60 120]

# 矩阵除法
c = a / b
c = np.divide(a,b)
print(c)
>>>

# 元素乘方
c = a**2
print(c)
>>>[ 100  400  900 1600]

# sin函数值
c=10*np.sin(a)
print(c)
>>> [-5.44021111  9.12945251 -9.88031624  7.4511316 ]

# 逻辑判断
c = b < 3
print(c)
>>> [ True  True  True False]
```

### 3.2 标准矩阵运算

```py
import numpy as np
a=np.array([[1,1],[0,1]])
b=np.arange(4).reshape((2,2))

# 点乘
c_dot = np.dot(a,b)
print(c_dot)
c_dot_2 = a.dot(b)
print(c_dot_2)
>>>[[2 4]
 [2 3]]

# sum,min,max函数的应用
a=np.random.random((2,4)) #生成0-1之间的随机矩阵
print(a)
>>>[[ 0.97930669  0.78582553  0.42964434  0.44356308]
 [ 0.44832963  0.89265819  0.41121829  0.78877376]]

np.sum(a)
>>>5.179319520751319

np.max(a)
>>>0.97930668689853739

np.min(a)
>>>0.41121828812772232
```

>如果你需要对行或者列进行查找运算，就需要在上述代码中为 axis 进行赋值。 当axis的值为0的时候，将会以列作为查找单元， 当axis的值为1的时候，将会以行作为查找单元。

```py
print("a =",a)
>>> [[ 0.97930669  0.78582553  0.42964434  0.44356308]
 [ 0.44832963  0.89265819  0.41121829  0.78877376]]

print(np.sum(a, axis=1))
>>> [ 2.63833965  2.54097987]

print(np.max(a, axis=0))
>>> [ 0.97930669  0.89265819  0.42964434  0.78877376]

print(np.min(a, axis=1))
>>> [ 0.42964434  0.41121829]
```

## 4. numpy基础运算2

```py
import numpy as np
A = np.arange(2,14).reshape((3,4))

# array([[ 2, 3, 4, 5]
#        [ 6, 7, 8, 9]
#        [10,11,12,13]])

# argmin, argmax找到最大最小元素下标
# 未指定axis时，将整个矩阵当做1维矩阵
# 可以指定axis
print(np.argmin(A))    # 0
>>> 0
print(np.argmin(A, axis=1))
>>> [0 0 0]

print(np.argmax(A))    # 11
>>> 11
print(np.argmax(A, axis=0))
>>> [2, 2, 2, 2]

# 统计均值
print(np.mean(A))        # 7.5
print(np.average(A))     # 7.5
print(A.mean())          # 7.5
>>> 7.5

# 求解中位数
print(np.median(A)))       # 7.5
>>> 7.5

# 在cumsum()函数中：生成的每一项矩阵元素均是从原矩阵首项累加到对应项的元素之和。比如元素9，在cumsum()生成的矩阵中序号为3，即原矩阵中2，3，4三个元素的和。
# 斐波拉契数列 累加运算函数
print(np.cumsum(A))
>>> [ 2  5  9 14 20 27 35 44 54 65 77 90]

# 累差运算函数 每一行中后一项与前一项之差
# 一个3行4列矩阵通过函数计算得到的矩阵便是3行3列的矩阵。
print(np.diff(A))
>>>[[1 1 1]
 [1 1 1]
 [1 1 1]]

# nonzero()函数
# 这个函数将所有非零元素的行与列坐标分割开，重构成两个分别关于行和列的矩阵。
print(np.nonzero(A))
>>>(array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]))

# 排序函数
# 可以指定axis 对不同维度进行排序
A = np.arange(14,2, -1).reshape((3,4))
print(A)
>>>[[14 13 12 11]
 [10  9  8  7]
 [ 6  5  4  3]]
print(np.sort(A))
>>>[[11 12 13 14]
 [ 7  8  9 10]
 [ 3  4  5  6]]
print(np.sort(A, axis=0)
>>>[[ 6  5  4  3]
 [10  9  8  7]
 [14 13 12 11]]

# 矩阵转置
print(np.transpose(A))
print(A.T)
>>>[[14 10  6]
 [13  9  5]
 [12  8  4]
 [11  7  3]]

# clip函数
# 将所有矩阵元素切割到指定范围
print(A)
>>>[[14 13 12 11]
 [10  9  8  7]
 [ 6  5  4  3]]
print(np.clip(A,5,9))
>>>[[9 9 9 9]
 [9 9 8 7]
 [6 5 5 5]]
```

## 5. numpy索引

### 5.1 一维索引

```py
import numpy as np
A = np.arange(3,15)

# array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
print(A[3])    # 6
>>> 6

A = np.arange(3,15).reshape((3,4))
"""
array([[ 3,  4,  5,  6]
       [ 7,  8,  9, 10]
       [11, 12, 13, 14]])
"""
print(A[2])
>>>[11 12 13 14]
```

### 5.2 二维索引

```py
# 访问具体元素
print(A[1][1])      # 8
print(A[1, 1])      # 8
>>> 8

# 在Python的 list 中，我们可以利用:对一定范围内的元素进行切片操作，在Numpy中我们依然可以给出相应的方法
print(A[1, 1:3])
>>> [8 9]

# flat flatten
# 这一脚本中的flatten是一个展开性质的函数，将多维的矩阵进行展开成1行的数列。而flat是一个迭代器，本身是一个object属性。
A = np.arange(3,15).reshape((3,4))
print(A.flatten())
# array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
>>>[ 3  4  5  6  7  8  9 10 11 12 13 14]

for item in A.flat:
    print(item)
>>>3
4
5
6
7
8
9
10
11
12
13
14
```

## 6. numpy 合并

```py
import numpy as np
A = np.array([1,1,1])
B = np.array([2,2,2])

# 按列合并 np.vstack()
print(np.vstack((A,B)))    # vertical stack
>>>[[1 1 1]
 [2 2 2]]

# 按行合并 np.hstack()
print(np.hstack((A, B)))
>>>[1 1 1 2 2 2]

# 新增维度 np.newaxis()
print(A[np.newaxis,:])
>>>[[1 1 1]]
print(A[:,np.newaxis])
>>>[[1]
 [1]
 [1]]

A = np.array([1,1,1])[:,np.newaxis]
B = np.array([2,2,2])[:,np.newaxis]
C = np.vstack((A,B))   # vertical stack
D = np.hstack((A,B))   # horizontal stack
print(C)
>>>[[1]
 [1]
 [1]
 [2]
 [2]
 [2]]
print(D)
>>>[[1 2]
 [1 2]
 [1 2]]

# 合并多个矩阵
C = np.concatenate((A,B,B,A),axis=0)
print(C)
>>>[[1]
 [1]
 [1]
 [2]
 [2]
 [2]
 [2]
 [2]
 [2]
 [1]
 [1]
 [1]]

D = np.concatenate((A,B,B,A),axis=1)
print(D)
>>>[[1 2 2 1]
 [1 2 2 1]
 [1 2 2 1]]
```

## 7. Numpy array 分割

```py
import numpy as np
A = np.arange(12).reshape((3, 4))
print(A)
>>>[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

# 纵向分割
print(np.split(A, 2, axis=1))
>>>[array([[0, 1],
       [4, 5],
       [8, 9]]),
    array([[ 2,  3],
       [ 6,  7],
       [10, 11]])]

# 横向分割
print(np.split(A, 3, axis=0))
>>>[array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]

# 错误分割
# print(np.split(A, 3, axis=1))
# ValueError: array split does not result in an equal division

# 不等量分割
print(np.array_split(A, 3, axis=1))
>>>[array([[0, 1],
       [4, 5],
       [8, 9]]), array([[ 2],
       [ 6],
       [10]]), array([[ 3],
       [ 7],
       [11]])]

# 在Numpy里还有np.vsplit()与横np.hsplit()方式可用。
print(np.vsplit(A, 3)) #等于 print(np.split(A, 3, axis=0))
>>> [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]

print(np.hsplit(A, 2)) #等于 print(np.split(A, 2, axis=1))
>>> [array([[0, 1],
       [4, 5],
       [8, 9]]), array([[ 2,  3],
       [ 6,  7],
       [10, 11]])]
```

## 8. Numpy copy & deep copy

### 8.1 = 的赋值方式会带有关联性

```py
import numpy as np

a = np.arange(4)
b = a
a[0] = 1
print(a, b)
>>>[1 1 2 3] [1 1 2 3]
```

### 8.2 copy() 的赋值方式没有关联性

```py
import numpy as np

a = np.arange(4)
b = a.copy()
a[0] = 1
print(a, b)
>>>[1 1 2 3] [0 1 2 3]
```