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

## 2.