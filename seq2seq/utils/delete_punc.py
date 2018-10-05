# coding=utf-8
import string
import sys
import os
import math
import re

ofs = open(sys.argv[3], 'w')
dictt = {}
for line in open(sys.argv[2]):
    line = line.strip()
    dictt[line] = 1 
for line in  open(sys.argv[1]):
    line = line.strip()
    strr = ''
    spe = ''
    splits = line.split('\t')
    if len(splits) != 2:
        continue
    ones1 = splits[0].split(' ')
    for one1 in ones1:
        if dictt.has_key(one1):
            continue
        strr += spe + one1
        spe = ' '
    strr += '\t'
    spe = ''
    ones2 = splits[1].split(' ')
    for one2 in ones2:
        if dictt.has_key(one2):
            continue
        strr += spe + one2
        spe = ' '
    strr = strr.strip('\t')
    ofs.write(strr+'\n')
ofs.close()
