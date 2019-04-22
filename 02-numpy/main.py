# encoding: utf-8
import numpy as np

def test_02():
    # 打印numpy的三种属性
    array = np.array([[1,2,3], [4,5,6]])
    print(array)
    print('ndim', array.ndim)
    print('shape', array.shape)
    print('size', array.size)

def test_03_01():
    import numpy as np
    a=np.array([10,20,30,40])   # array([10, 20, 30, 40])
    b=np.arange(4)              # array([0, 1, 2, 3])
    # 矩阵减法
    c=a-b  # array([10, 19, 28, 37])
    c = np.subtract(a, b)
    print(c)

    # 矩阵加法
    c = a + b
    c = np.add(a, b)
    print(c)

    # 矩阵乘法
    c = a * b
    c = np.multiply(a, b)
    print(c)

    # 矩阵除法
    c = a / b
    c = np.divide(a,b)
    print(c)

    # 元素乘方
    c = a**2
    print(c)

    # sin函数值
    c=10*np.sin(a)
    print(c)

    # 逻辑判断
    c = b < 3
    print(c)

def test_03_02():
    import numpy as np
    a=np.array([[1,1],[0,1]])
    b=np.arange(4).reshape((2,2))

    # 点乘
    c_dot = np.dot(a,b)
    print(c_dot)
    c_dot_2 = a.dot(b)
    print(c_dot_2)

    # sum,min,max函数的应用
    a=np.random.random((2,4)) #生成0-1之间的随机矩阵
    print(a)
    print(np.sum(a))
    print(np.max(a))
    print(np.min(a))

    print("a =",a)
    print(np.sum(a, axis=1))
    print(np.max(a, axis=0))
    print(np.min(a, axis=1))

def test_04():
    import numpy as np
    A = np.arange(2,14).reshape((3,4))

    # array([[ 2, 3, 4, 5]
    #        [ 6, 7, 8, 9]
    #        [10,11,12,13]])

    # argmin, argmax找到最大最小元素下标
    # 未指定axis时，将整个矩阵当做1维矩阵
    # 可以指定axis
    print(np.argmin(A))    # 0
    print(np.argmin(A, axis=1))

    print(np.argmax(A))    # 11
    print(np.argmax(A, axis=0))

    # 统计均值
    print(np.mean(A))        # 7.5
    print(np.average(A))     # 7.5
    print(A.mean())          # 7.5

    # 求解中位数
    print(np.median(A))

    # 在cumsum()函数中：生成的每一项矩阵元素均是从原矩阵首项累加到对应项的元素之和。比如元素9，在cumsum()生成的矩阵中序号为3，即原矩阵中2，3，4三个元素的和。
    # 斐波拉契数列 累加运算函数
    print(np.cumsum(A))

    # 累差运算函数 每一行中后一项与前一项之差
    # 一个3行4列矩阵通过函数计算得到的矩阵便是3行3列的矩阵。
    print(np.diff(A))

    # nonzero()函数
    # 这个函数将所有非零元素的行与列坐标分割开，重构成两个分别关于行和列的矩阵。
    print(np.nonzero(A))

    # 排序函数
    # 可以指定axis 对不同维度进行排序
    A = np.arange(14,2, -1).reshape((3,4))
    print(np.sort(A))
    print(np.sort(A, axis=0))

    # 矩阵转置
    print(np.transpose(A))
    print(A.T)

    # clip函数
    # 将所有矩阵元素切割到指定范围
    print(A)
    print(np.clip(A,5,9))

if __name__ == "__main__":
    test_03_02()