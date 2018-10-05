import pprint
import cPickle
import numpy as np
import gzip

pp = pprint.PrettyPrinter()

def save_pkl(path, obj):
    with open(path, 'w') as f:
        cPickle.dump(obj, f)
        print(" [*] save %s" % path)

def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
    print(" [*] load %s" % path)
    return obj

def save_npy(path, obj):
    np.save(path, obj)
    print(" [*] save %s" % path)

def load_npy(path):
    obj = np.load(path)
    print(" [*] load %s" % path)
    return obj

def prepare_data(raw_x, seq_len):
    '''
    >>> xx = [[1,], [2,3]]
    >>> x, m, l = prepare_data(xx, 3)
    >>> print(x)
    [[1 0 0]
     [2 3 0]]
    >>> x, m, l = prepare_data(xx, 2)
    >>> print(x)
    [[1 0]
     [2 0]]
    '''
    x = np.zeros([len(raw_x), seq_len], dtype='int32')
    m = np.zeros([len(raw_x), seq_len], dtype='float32')
    l = np.zeros([len(raw_x)], dtype='int32')
    for idx in range(len(raw_x)):
        l[idx] = min(seq_len, len(raw_x[idx]) + 1) # 1 for EOS
        m[idx, :l[idx]] = 1
        x[idx, :l[idx]-1] = raw_x[idx][:l[idx]-1]
    return x, m, l

if __name__ == '__main__':
    #import doctest
    #doctest.testmod()
    test_query = [[1,3,5,7,9]]
    x,m,l = prepare_data(test_query, 10)
    print("x:",x)
    print("m:",m)
    print("l:",l)
