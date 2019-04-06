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
    
    # for key in c.keys():
    #     print(key)
    # for value in c.values():
    #     print(value)
    # for key in c:
    #     print(key)
    # for key in c.items():
    #     print(key)
    # for key, value in c.items():
    #     print(key, value)
    for index, value in enumerate(a):
        print(index, value)
    for value in enumerate(a):
        print(value)
    for value in zip(a, b):
        print(value)
     
test_enumerate()
