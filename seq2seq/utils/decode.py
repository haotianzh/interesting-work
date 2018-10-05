# coding=utf-8
import string
import sys
import os
import math
import re

ofs = open(sys.argv[2], 'w')
for line in  open(sys.argv[1]):
    line = line.strip()
    line_unic = unicode(line,'gbk',errors='ignore')
    line_utf8 = line_unic.encode('utf8')
    ofs.write(line_utf8+'\n')
ofs.close()
