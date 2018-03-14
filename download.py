# -*- coding: utf-8 -*-
#

import os
import argparse
import subprocess

# Home: http://yann.lecun.com/exdb/mnist/
mnist_urls = [
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='mnist')
    args = p.parse_args()
    return args

def get_mnist():
    os.makedirs("mnist", exist_ok=True)
    os.chdir("mnist")
    for url in mnist_urls:
        fname = os.path.basename(url)
        ungzip_fname, _ = os.path.splitext(fname)
        if os.path.exists(ungzip_fname):
            continue
        if not os.path.exists(fname):
            ret = subprocess.check_call(["wget", "-c", url])
        ret = subprocess.check_call(["gzip", "-d", fname])

def main():
    args = get_args()
    if args.dataset == "mnist":
        get_mnist()

if __name__ == '__main__':
    main()
