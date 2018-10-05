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
    #parser.add_argument('--save_dir',  type=str, default='../results/child/gru/',
    #parser.add_argument('--save_dir',  type=str, default='../results/child/gru_multi_gpu/',
    #parser.add_argument('--save_dir',  type=str, default='../results/child/reduce_common_reply/',
    parser.add_argument('--save_dir',  type=str, default='../results/child/1200w/',
            help='model directory to store checkpointed models')
    # todo
    parser.add_argument('--beam_size', type=int, default=20,
            help='size for beam search')
    parser.add_argument('--max_seq_len', type=int, default=50,
            help='max sequence length for generation')
    parser.add_argument('--device', type=str, default='/cpu:0',
            help='device for generation')
    parser.add_argument('--idf', type=str, default='../data/child.idf',
            help='idf dict')
    parser.add_argument('--black', type=str, default='../data/black.reply',
            help='idf dict')
    args = parser.parse_args()
    return args

def print_predict(sents,scores,idx2word,rank_type):
    sents_tmp = []
    for idx,i in enumerate(sents):
        #skip unknown
        if 1 in i:
            continue
        sents_tmp.append(i)
    sents =  sents_tmp
    sents = [[idx2word[i].encode("utf8", "ignore") for i in s] for s in sents]

    ns = []
    for idx, s in enumerate(sents):
        tmp = ''
        for w in s:
            if w.strip() != "":
                tmp = tmp + w.strip()
        tmp = tmp.strip()
        if tmp not in ns:
            ns.append(tmp.strip())

    for idx, s in enumerate(ns):
        print("\t" + s, " : ", scores[idx])
    return ns


def load_idf(idf_file_path):
    if os.path.exists(idf_file_path) == False:
        print >>sys.stderr, "No such a idf file:", idf_file_path
        return None

    ret_idf_dict = {}
    for line in open(idf_file_path):
        infos = line.strip().split('\t')
        if len(infos) != 2:
            continue
        ret_idf_dict[infos[0]] = float(infos[1])
    print("idf dict size:", len(ret_idf_dict))
    return ret_idf_dict


# sents is unicode
def rerank_by_idf(idf_dict, sents):
    if idf_dict == None:
        return None
    sent_w_lst = []
    for s in sents:
        score = 0.0
        sent = ""
        valid_token_num = 0
        for w in s:
            if w.strip() == "":
                continue
            valid_token_num += 1
            score += idf_dict.get(w.encode('utf8', 'ignore'), 16)
            sent += w.encode('utf8', 'ignore')
        if valid_token_num == 0:
            continue
        score = score/valid_token_num
        sent_w_lst.append((sent, score))
    sent_w_lst.sort(key = lambda x:x[1], reverse = True)
    return sent_w_lst


def sample(args):
    print(args)
    black_reply_dict = {}
    for line in open(args.black):
        black_reply_dict[line.strip().split("\t")[0].strip()] = 1
    print("black_reply_dict size:", len(black_reply_dict))

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
                start = time.time()
                query = query.strip()

                str_in = query
                x = str_in.split()
                for w in x:
                    print(w),
                    print(w.decode('utf-8') in vocab)
                x = [[vocab.get(w.decode('utf-8'), 1) for w in x]]
                x, mx, lx = prepare_data(x, saved_args.seq_len)
                print(x)

                #lx_in = np.asarray([len(wx_in)], dtype='int32')
                sents, scores, common_sents, common_scores = m.generate_original(sess, vocab, sym_x, sym_lx, sym_y, x, lx, \
                        args.beam_size, args.max_seq_len, idx2word, black_reply_dict)
                print("query:", query)
                print("--"*10, "generated reply", "--"*10)
                ss = print_predict(sents, scores, idx2word, 1)
                #for s in ss:
                #    s = s.strip().encode("utf8", "ignore")
                #    if s in black_reply_dict:
                #        continue
                #    else:
                #        print("\t%s" % (s))
                print("--"*10, "common replys", "--"*10)
                ss = print_predict(common_sents, common_scores, idx2word, 1)


                finish = time.time()
                generateTime = (finish - start)
                print("generating cost time :", generateTime, " ms")


if __name__=='__main__':
    args = init_args()
    sample(args)
