# encoding: utf-8
def test_paser():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("echo")
    parser.add_argument('-f', '--fun', type=int, help='print(fun!')
    parser.add_argument('-a', action="store_true")
    parser.add_argument('-b', type=int, choices=[0, 1, 2])
    args = parser.parse_args()
    if(args.fun == 1):
        print("fun")

def test_enumerate():
    a = [1, 2, 3, 4, 5, 6]
    b = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
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
    for index, value in enumerate(a):
        print(index, value)
    for value in enumerate(a):
        print(value)
    for value in zip(a, b):
        print(value)

def expand_dim_test():
    import tensorflow as tf
    # t = tf.Variable([[1, 2],[3, 4]])
    t = tf.Variable([1, 2, 3, 4])    
    t1 = tf.expand_dims(t, 0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(t1))
     
def matrix_mul_test():
    import numpy as np
    m1 = np.ones((1,2,5))
    m2 = np.ones((5,5))
    m = np.dot(m1, m2)

    print(m.shape)

matrix_mul_test()
