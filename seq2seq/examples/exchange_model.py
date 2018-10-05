# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
import argparse
import time
import datetime
import os
import sys
from six.moves import cPickle as pkl

from utils.douban_reader import DoubanReader
from models.rnn import RnnAutoencoder
from utils.misc import *

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',  type=str, default='../results/child/1200w/',
            help='model directory to store checkpointed models')
    parser.add_argument('--device', type=str, default='/cpu:0',
            help='device for generation')
    args = parser.parse_args()
    return args


model_dir = "model_pb"

def reload(args):

    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pkl.load(f)
    with open(os.path.join(args.save_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pkl.load(f)
        idx2word = {v: k for (k, v) in vocab.items()}
        idx2word[0] = '\n'
        idx2word[1] = '<UNK>'
        with open("%s/rnn.vcb" % (model_dir), "w") as vcb_f:
            for (k, v) in vocab.items():
                vcb_f.write(k.encode("utf8", "ignore") + "\t" + str(v) + "\n")


    g1 = tf.Graph()
    with tf.device(args.device), g1.as_default():
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
    
        #sym_x, sym_lx, sym_y = inputs[0], inputs[1], inputs[2]
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver(tf.all_variables())
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)


            save_vars = {}
            for v in tf.trainable_variables():
                save_vars[v.value().name] = sess.run(v)
            g2 = tf.Graph()
            with g2.as_default():
                consts = {}
                for k in save_vars.keys():
                    consts[k] = tf.constant(save_vars[k])
                tf.import_graph_def(g1.as_graph_def(), input_map={name:consts[name] for name in consts.keys()})
                tf.train.write_graph(g2.as_graph_def(), model_dir, 'rnn.pb' , False)
                tf.train.write_graph(g2.as_graph_def(), model_dir, 'rnn.txt')

    

if __name__=='__main__':
    args = init_args()
    reload(args)
