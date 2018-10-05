#!/usr/bin/env python
# encoding: utf-8

import sys
import math

def load_file(file_path):
    voc = {}
    line_num = 0
    with open(file_path, 'r') as file:
        for line in file:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            word_list = set(line.split(' '))
            for word in word_list:
                if word not in voc:
                    voc[word] = 1
                else:
                    voc[word] += 1
    # print line_num
    return voc, line_num

def chi_square(related_data_dict, related_line_num, not_related_data_dict, \
              not_related_line_num, total_data_dict, total_line_num):
    chisquare_dict = {}
    #print "word\tchi square\tidf"
    for word in related_data_dict:
        if word not in not_related_data_dict:
            continue
        # print word
        # print related_data_dict[word]
        # print not_related_data_dict[word]
        # print related_line_num - related_data_dict[word]
        # print not_related_line_num - not_related_data_dict[word]
        numerator = float(related_line_num + not_related_line_num) * \
                    pow(float(related_data_dict[word] * \
                              (not_related_line_num - \
                               not_related_data_dict[word]) -
                              not_related_data_dict[word] * \
                              (related_line_num - related_data_dict[word])), 2)
        denominator = related_line_num * \
                      (related_data_dict[word] + not_related_data_dict[word]) * \
                      not_related_line_num * \
                      (not_related_line_num - not_related_data_dict[word] + \
                       related_line_num - related_data_dict[word])
        chi_square_idf_dict = {}
        chi_square_idf_dict[numerator/denominator] = math.log(\
            float(total_line_num)/total_data_dict[word])
        chisquare_dict[word] = chi_square_idf_dict
    return chisquare_dict


def main():
    if len(sys.argv) != 3:
        print "Invalid Format. Usage: python " + sys.argv[0] + \
              " PostiveFile NegativeFile"
        return
    related_data_dict, related_line_num = load_file(sys.argv[1])
    not_related_data_dict, not_related_line_num = load_file(sys.argv[2])
    total_line_num = related_line_num + not_related_line_num
    total_data_dict = {}
    for word in related_data_dict:
        if word in not_related_data_dict:
            total_data_dict[word] = related_data_dict[word] + \
                                    not_related_data_dict[word]
        else:
            total_data_dict[word] = related_data_dict[word]
    for word in not_related_data_dict:
        if word not in related_data_dict:
            total_data_dict[word] = not_related_data_dict[word]
    chisquare_dict = chi_square(related_data_dict, related_line_num, \
                                not_related_data_dict, not_related_line_num, \
                                total_data_dict, total_line_num)

    chi_square_value_tuple = sorted(chisquare_dict.items(),
                    key=lambda chisquare_dict : chisquare_dict[1],
                    reverse=True)
    print "word\tchi square\tidf"
    for e in chi_square_value_tuple:
        print "%s\t%s\t%f" % (e[0], e[1].keys()[0], e[1][e[1].keys()[0]])

if  __name__ == "__main__":
    main()
