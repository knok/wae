# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import configs
from wae import WAE
from datahandler import DataHandler
import utils
import numpy as np
import matplotlib.pyplot as plt

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

    data = DataHandler(opts)
    wae = WAE(opts)
    wae.restore_checkpoint(FLAGS.checkpoint)

    batch_img = data.data[0:2]
    enc_vec = wae.sess.run(wae.encoded,
                           feed_dict={wae.sample_points: batch_img,
                                      wae.is_training: False})
    vdiff = enc_vec[1] - enc_vec[0]
    vdiff = vdiff / 10
    gen_vec = np.zeros((10, vdiff.shape[0]), dtype=np.float32)
    for i in range(10):
        gen_vec[i, :] = enc_vec[0] + vdiff * i

    sample_gen = wae.sess.run(wae.decoded,
                              feed_dict={wae.sample_noise: gen_vec,
                                         wae.is_training:False})
    img = np.hstack(sample_gen)
    img = (img + 1.0) / 2
    plt.imshow(img)
    plt.savefig('analogy.png')
    #import pdb; pdb.set_trace()
    #pass

if __name__ == '__main__':
    main()


