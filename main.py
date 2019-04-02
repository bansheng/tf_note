# encoding: utf-8
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("echo")
parser.add_argument('-f', '--fun', type=int, help='print(fun!')
parser.add_argument('-a', action="store_true")
parser.add_argument('-b', type=int, choices=[0, 1, 2])
args = parser.parse_args()
if(args.fun == 1):
    print("fun")

