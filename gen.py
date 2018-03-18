# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import configs
from wae import WAE
from datahandler import DataHandler
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--exp')
parser.add_argument('--zdim', type=int)
parser.add_argument('--checkpoint', '-c', type=str)

FLAGS = parser.parse_args()

def main():
    if FLAGS.exp == 'dir64':
        opts = configs.config_dir64
    else:
        assert False, 'Unknown experiment configuration'

    if FLAGS.zdim is not None:
        opts['zdim'] = FLAGS.zdim

    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    wae = WAE(opts)
    wae.restore_checkpoint(FLAGS.checkpoint)
    import pdb; pdb.set_trace()
    pass

if __name__ == '__main__':
    main()


