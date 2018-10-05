# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf

import argparse
import time
import os
import sys
from six.moves import cPickle as pkl

from utils.douban_reader import DoubanReader
from models.rnn_debug import RnnAutoencoder
from utils.misc import *

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',  type=str, default='../results/child/toy/',
            help='model directory to store checkpointed models')
    # todo
    parser.add_argument('--beam_size', type=int, default=20,
            help='size for beam search')
    parser.add_argument('--max_seq_len', type=int, default=50,
            help='max sequence length for generation')
    parser.add_argument('--device', type=str, default='/cpu:0',
            help='device for generation')
    args = parser.parse_args()
    return args

def sample(args):
    print(args)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pkl.load(f)
    with open(os.path.join(args.save_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pkl.load(f)
        idx2word = {v: k for (k, v) in vocab.items()}
        idx2word[0] = '\n'
        idx2word[1] = '<UNK>'
    
    with tf.device(args.device):
        m = RnnAutoencoder(rnn_type=saved_args.rnn_type,
                       batch_size=saved_args.batch_size,
                       dim_emb=saved_args.dim_emb,
                       num_units=saved_args.num_units,
                       vocab_size=saved_args.vocab_size,
                       seq_len=saved_args.seq_len,
                       grad_clip=saved_args.grad_clip,
                       learning_rate=saved_args.learning_rate,
                       infer=True)

        inputs = m.get_inputs(infer=True)
        m.build_model(inputs, infer=True)

    sym_x, sym_lx, sym_y = inputs[0], inputs[1], inputs[2]
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model load done")

            while 1:
                query = sys.stdin.readline()
                if not query:
                    break
                query = query.strip()

                str_in = query
                x = str_in.split()
                #for w in x:
                #    print(w),
                #    print(w.decode('utf-8') in vocab)
                x = [[vocab.get(w.decode('utf-8'), 0) for w in x]]
                x, mx, lx = prepare_data(x, saved_args.seq_len)
                #print(x)

                #lx_in = np.asarray([len(wx_in)], dtype='int32')
                sents, scores = m.generate(sess, vocab, sym_x, sym_lx, sym_y, x, lx, \
                        args.beam_size, args.max_seq_len, idx2word)
                sents = [[idx2word[i] for i in s] for s in sents]
                ns = []
                for idx,s in enumerate(sents):
                    tmp = ''
                    for w in s:
                        tmp = tmp + w
                    tmp = tmp.strip()
                    #ns.append(tmp.strip())
                    if tmp not in ns:
                        ns.append(tmp.strip().encode('utf-8', 'ingore'))
                print ("query:", query)
                for idx, s in enumerate(ns):
                    print(s)
                    print("\tscores:", scores[idx])


if __name__=='__main__':
    args = init_args()
    sample(args)
