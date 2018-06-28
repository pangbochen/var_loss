import torch
import os
import sys, getopt

for i in range(1, len(sys.argv)):
    print(i,sys.argv[i])

opts, args = getopt.getopt(sys.argv[1:], '', ["vis_env="])
for op, value in opts:
    if op == '--vis_env':
        print('vis_env is {}'.format(value))