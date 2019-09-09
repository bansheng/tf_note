# numpy函数

## 1. np.prod

```py
numpy.prod(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>)[source]
Return the product of array elements over a given axis.
```

返回矩阵元素相乘的结果，可以指定坐标轴

```py
# By default, calculate the product of all elements:
>>>
>>> np.prod([1.,2.])
2.0

# Even when the input array is two-dimensional:
>>>
>>> np.prod([[1.,2.],[3.,4.]])
24.0

# But we can also specify the axis over which to multiply:
>>>
>>> np.prod([[1.,2.],[3.,4.]], axis=1)
array([  2.,  12.])
```

## 2. np.argsort

从中可以看出argsort函数返回的是数组值从小到大的索引值

```python
#     One dimensional array:一维数组
    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])
#     Two-dimensional array:二维数组
    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])
    >>> np.argsort(x, axis=0) #按列排序
    array([[0, 1],
           [1, 0]])
    >>> np.argsort(x, axis=1) #按行排序
    array([[0, 1],
           [0, 1]])
```

## 3. np.bincount

```python
np.bincount(x, weights=None, minlength=None)
```

它大致说bin的数量比x中的最大值大1，每个bin给出了它的索引值在x中出现的次数。

```python
# 我们可以看到x中最大的数为7，因此bin的数量为8，那么它的索引值为0->7
x = np.array([0, 1, 1, 3, 2, 1, 7])
# 索引0出现了1次，索引1出现了3次......索引5出现了0次......
>>> np.bincount(x)
array([1, 3, 1, 1, 0, 0, 0, 1])

# 我们可以看到x中最大的数为7，因此bin的数量为8，那么它的索引值为0->7
>>> x = np.array([7, 6, 2, 1, 4])
# 索引0出现了0次，索引1出现了1次......索引5出现了0次......
>>> np.bincount(x)
array([0, 1, 1, 0, 1, 0, 1, 1])
```

如果weights参数被指定，那么x会被它加权，也就是说，如果值n发现在位置i，那么out[n] += weight[i]而不是out[n] += 1.**因此，我们weights的大小必须与x相同，否则报错。**

```python
w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6])
# 我们可以看到x中最大的数为4，因此bin的数量为5，那么它的索引值为0->4
x = np.array([2, 1, 3, 4, 4, 3])
# 索引0 -> 0
# 索引1 -> w[1] = 0.5
# 索引2 -> w[0] = 0.3
# 索引3 -> w[2] + w[5] = 0.2 - 0.6 = -0.4
# 索引4 -> w[3] + w[4] = 0.7 + 1 = 1.7
>>> np.bincount(x,  weights=w)
array([ 0. ,  0.5,  0.3, -0.4,  1.7])
```

我们来看一下minlength这个参数。文档说，如果minlength被指定，那么输出数组中bin的数量至少为它指定的数。

```python
# 我们可以看到x中最大的数为3，因此bin的数量为4，那么它的索引值为0->3
>>> x = np.array([3, 2, 1, 3, 1])
# 本来bin的数量为4，现在我们指定了参数为7，因此现在bin的数量为7，所以现在它的索引值为0->6
>>> np.bincount(x, minlength=7)
array([0, 2, 1, 2, 0, 0, 0])

# 我们可以看到x中最大的数为3，因此bin的数量为4，那么它的索引值为0->3
>>> x = np.array([3, 2, 1, 3, 1])
# 本来bin的数量为4，现在我们指定了参数为1，那么它指定的数量小于原本的数量，因此这个参数失去了作用，索引值还是0->3
>>> np.bincount(x, minlength=1)
array([0, 2, 1, 2])
```

## 4. np.std(x, ddof=0)

numpy.std() 求标准差的时候默认是除以 n 的，即是有偏的，np.std无偏样本标准差方式为加入参数 ddof = 1；
pandas.std() 默认是除以n-1 的，即是无偏的，如果想和numpy.std() 一样有偏，需要加上参数ddof=0 ，即pandas.std(ddof=0) ；DataFrame的describe()中就包含有std()；

```python
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.std(a, ddof = 1)
3.0276503540974917
>>> np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
3.0276503540974917
>>> np.sqrt(( a.var() * a.size) / (a.size - 1))
3.0276503540974917
```

## 5. np.max 和 np.maximum的区别

1. 参数
    首先比较二者的参数部分：

    ```python
    np.max：(a, axis=None, out=None, keepdims=False)
    # 求序列的最值
    # 最少接收一个参数
    # axis：默认为列向（也即 axis=0），axis = 1 时为行方向的最值；
    np.maximum：(X, Y, out=None)
    # X 与 Y 逐位比较取其大者；
    # 最少接收两个参数
    ```

2. 使用上

    ```python
    >> np.max([-2, -1, 0, 1, 2])
    2
    >> np.maximum([-2, -1, 0, 1, 2], 0)
    array([0, 0, 0, 1, 2])
    # 当然 np.maximum 接受的两个参数，也可以大小一致
    # 或者更为准确地说，第二个参数只是一个单独的值时，其实是用到了维度的 broadcast 机制；
    ```

## 6. np.random.choice(a, size=None, replace=True, p=None)

- replace 为true代表可能有重复，false代表不重复
- a可以是单独数字代表range(a)，也可以是一维数组
- p代表a中选项对应概率，None代表均匀分布

## 7. numpy.squeeze()函数

语法：numpy.squeeze(a,axis = None)

 1）a表示输入的数组；
 2）axis用于指定需要删除的维度，但是指定的维度值必须为1，否则将会报错；
 3）axis的取值可为None 或 int 或 tuple of ints, 可选。若axis为空，则删除所有单维度的条目；
 4）返回值：数组
 5) 不会修改原数组；

```python
>>> c  = np.arange(10).reshape(2,5)
>>> c
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
>>> c.squeeze()
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
>>> d = c
>>> d = d.reshape([1,10])
>>> d
array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
>>> np.squeeze(d)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

## 8. np.ravel()

与np.flatten()相同，但是返回的是引用，会对原数据产生影响

```python
>>> a = np.array([[1,2],[3,4],[5,6]])
>>> a
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> b = a.flatten()
>>> b
array([1, 2, 3, 4, 5, 6])
>>> c = a.ravel()
>>> c
array([1, 2, 3, 4, 5, 6])
>>> c[3] = 100
>>> a
array([[  1,   2],
       [  3, 100],
       [  5,   6]])
>>> b
array([1, 2, 3, 4, 5, 6])
>>> c
array([  1,   2,   3, 100,   5,   6])
```

## 9. numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)

the following norms can be calculated:

ord | norm for matrices | norm for vectors
:-: | :-: | :-:
None | Frobenius norm | 2-norm
‘fro’ | Frobenius norm | –
‘nuc’ | nuclear norm | –
inf | max(sum(abs(x), axis=1)) | max(abs(x))
-inf | min(sum(abs(x), axis=1)) | min(abs(x))
0 | – | sum(x != 0)
1 | max(sum(abs(x), axis=0)) | as below
-1 | min(sum(abs(x), axis=0)) | as below
2 | 2-norm (largest sing. value) | as below
-2 | smallest singular value | as below
other | – | sum(abs(x)**ord)**(1./ord)

```python
>>> from numpy import linalg as LA
>>> a = np.arange(9) - 4
>>> a
array([-4, -3, -2, ...,  2,  3,  4])
>>> b = a.reshape((3, 3))
>>> b
array([[-4, -3, -2],
       [-1,  0,  1],
       [ 2,  3,  4]])
>>> LA.norm(a)
7.745966692414834
>>> LA.norm(b)
7.745966692414834
>>> LA.norm(b, 'fro')
7.745966692414834
>>> LA.norm(a, np.inf)
4.0
>>> LA.norm(b, np.inf)
9.0
>>> LA.norm(a, -np.inf)
0.0
>>> LA.norm(b, -np.inf)
2.0
```

## 10.np.add.at(dw, indice, 1)

```shell
In [226]: x = [[0,4,1], [3,2,4]]
     ...: dW = np.zeros((5,6),int)

In [227]: np.add.at(dW,x,1)
In [228]: dW
Out[228]:
array([[0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0]])
```
