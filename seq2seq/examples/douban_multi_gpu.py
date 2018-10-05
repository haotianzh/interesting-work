import sys
sys.path.append('../')

import os
import cPickle as pkl
import argparse
import time
import numpy as np

import tensorflow as tf

from utils.douban_reader import DoubanReader
from utils.batch_iterator import BatchIterator
from utils.misc import *
from models.rnn import RnnAutoencoder


#gpus = [1, 2, 3]
gpus = [0, 1, 2]

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_type', type=str, default='gru',
            help='rnn_type in [\'lstm\', \'gru\']')
    parser.add_argument('--init_from', type=str, default=None,
            help='the directory with trained model')
    parser.add_argument('--data_dir', type=str, default='../data/train_data', \
            help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='../results/child/gru_multi_gpu', \
            help='directory to store the checkpointed models')
    parser.add_argument('--dim_emb', type=int, default=300, \
            help='dimension for word embedding')
    parser.add_argument('--num_units', type=int, default=512, \
            help='number of units for rnn')
    parser.add_argument('--num_epochs', type=int, default=100, \
            help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32*len(gpus), \
            help='size of each batch')
    parser.add_argument('--learning_rate', type=float, default=0.0005, \
            help='learning rate')
    parser.add_argument('--grad_clip', type=float, default=5, \
            help='clipping the gradient')
    parser.add_argument('--valid_period', type=int, default=5, \
            help='get the results from valid set between n epochs')
    parser.add_argument('--test_period', type=int, default=5, \
            help='get the results from test set between n epochs')
    parser.add_argument('--save_every', type=int, default=1000,
            help='save the model at every n batches')
    parser.add_argument('--seq_len', type=int, default=50, \
            help='the maximal sequence length, longer will be truncted')
    #parser.add_argument('--num_gpus', type=int, default='2',
            #help='default number of gpu to use')
    args = parser.parse_args()

    print(args)
    return args


def average_gradients(tower_grads):
    with tf.name_scope('average_gradients'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def train(args):
    data_reader = DoubanReader(args.data_dir)
    args.vocab_size = data_reader.vocab_size
    print(args.vocab_size)

    if args.init_from is not None:
        assert os.path.isdir(args.init_from), "%s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, 'config.pkl')), \
                'config.pkl file does not exist in path %s' % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, 'vocab.pkl')), \
                'vocab.pkl file does not exist in path %s' % args.init_from
        ckpt =  tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'

        with open(os.path.join(args.init_from, 'config.pkl')) as f:
            saved_model_args = pkl.load(f)
        need_to_be_same = ['rnn_type', 'num_units', 'dim_emb',]
        for checkme in need_to_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme], \
                    'Command line argument and saved model disagree on %s' % checkme

    it_train = BatchIterator(len(data_reader.train_data[0]), args.batch_size, 
                             data_reader.train_data, testing=False)
    it_valid = BatchIterator(len(data_reader.valid_data[0]), args.batch_size, 
                             data_reader.valid_data, testing=False)
    it_test = BatchIterator(len(data_reader.test_data[0]), args.batch_size, 
                             data_reader.test_data, testing=False)

    num_batches_train = len(data_reader.train_data[0]) / args.batch_size
    num_batches_valid = len(data_reader.valid_data[0]) / args.batch_size
    num_batches_test = len(data_reader.test_data[0]) / args.batch_size

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pkl.dump(args, f)
    with open(os.path.join(args.save_dir, 'vocab.pkl'), 'wb') as f:
        pkl.dump(data_reader.vocab, f)
    
    with tf.Graph().as_default():
        # todo: sgd?
        opt = tf.train.AdamOptimizer(args.learning_rate)
        m = RnnAutoencoder(rnn_type=args.rnn_type,
                       batch_size=args.batch_size/len(gpus),
                       dim_emb=args.dim_emb,
                       num_units=args.num_units,
                       vocab_size=args.vocab_size,
                       seq_len=args.seq_len,
                       grad_clip=args.grad_clip,
                       learning_rate=args.learning_rate,
                       infer=False,)
                
        # get the input format, the batchsize is default set to None
        sx, slx, sy, smy = m.get_inputs(infer=False)
            
        # init the variable on cpu0, although redundant operations are also introduced.
        with tf.device('/cpu:0'):
            with tf.name_scope('cpu_aux'):
                m.build_model([sx, slx, sy, smy], infer=False)
                tf.get_variable_scope().reuse_variables()
            
        tower_grads = []
        total_cost = []
        for i in range(len(gpus)):
            with tf.device('/gpu:%d' % gpus[i]):
                print("USE GPU ID:",gpus[i])
                with tf.name_scope('%s_%d' % (m.TOWER_NAME, i)) as scope:
                    # split the input for each device
                    x_slice = tf.gather(sx, range(i*args.batch_size/len(gpus), (i+1)*args.batch_size/len(gpus)))
                    lx_slice = tf.gather(slx, range(i*args.batch_size/len(gpus), (i+1)*args.batch_size/len(gpus)))
                    y_slice = tf.gather(sy, range(i*args.batch_size/len(gpus), (i+1)*args.batch_size/len(gpus)))
                    my_slice = tf.gather(smy, range(i*args.batch_size/len(gpus), (i+1)*args.batch_size/len(gpus)))
                    input_slice = [x_slice, lx_slice, y_slice, my_slice]
    
                    cost = m.build_model(input_slice, infer=False)
                    tf.get_variable_scope().reuse_variables()
    
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    grads = opt.compute_gradients(cost)
                    tower_grads.append(grads)
                    total_cost.append(cost)
    
        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads)
        total_cost = tf.add_n(total_cost)/len(gpus)
            
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))
    
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_summary(summaries)
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(init)
        #tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.train.SummaryWriter(args.save_dir, sess.graph)
    
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
    
        for e in range(args.num_epochs):
            # train
            outs = []
            cnt_words = 0
            for b in range(num_batches_train):
                x, y = it_train.next()
                x, mx, lx = prepare_data(x, args.seq_len)
                y, my, ly = prepare_data(y, args.seq_len)
                out, _ = sess.run([total_cost, apply_gradient_op], {sx: x, slx: lx, sy: y, smy: my,})
                outs.append(out)
                cnt_words += np.sum(ly)
    
                # save the model
                if (e*num_batches_train + b + 1) % args.save_every == 0 \
                        or (e == args.num_epochs - 1 and b == num_batches_train - 1):
                    print('Save at step {}: {:.3f}'.format(e*num_batches_train + b,
                        np.exp(np.sum(outs)*args.batch_size/cnt_words)))
                    checkpoint_path = os.path.join(args.save_dir, 'model_ckpt')
                    saver.save(sess, checkpoint_path, global_step = e*num_batches_train + b)
            print('Epoch {}: train loss {}'.format(e, np.exp(np.sum(outs)*args.batch_size/cnt_words)))
            
            # valid
            outs = []
            cnt_words = 0
            for b in range(num_batches_valid):
                x, y = it_valid.next()
                x, mx, lx = prepare_data(x, args.seq_len)
                y, my, ly = prepare_data(y, args.seq_len)
                out, = sess.run([cost,], {sx: x, slx: lx, sy: y, smy: my,})
                #print(np.exp(out*args.batch_size/np.sum(ly)))
                outs.append(out)
                cnt_words += np.sum(ly)
            print('Epoch {}: valid loss {}'.format(e, np.exp(np.sum(outs)*args.batch_size/cnt_words)))
    
            # test
            outs = []
            cnt_words = 0
            for b in range(num_batches_test):
                x, y = it_test.next()
                x, mx, lx = prepare_data(x, args.seq_len)
                y, my, ly = prepare_data(y, args.seq_len)
                out, = sess.run([cost,], {sx: x, slx: lx, sy: y, smy: my,})
                outs.append(out)
                cnt_words += np.sum(ly)
            print('Epoch {}: test loss {}'.format(e, np.exp(np.sum(outs)*args.batch_size/cnt_words)))
    
if __name__ == '__main__':
    args = load_args()
    train(args)
