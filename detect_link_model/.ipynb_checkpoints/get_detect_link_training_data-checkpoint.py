"""input: text data (wiki dump), one line means one article (.txt)
output: training data (.json) with features (link probability, frequency, first occurrence, last occurrence, spread)
"""

import re
import json
from bs4 import BeautifulSoup
from nltk import ngrams, stem
from math import log
from copy import deepcopy
import pandas as pd
import sys
sys.path.append("..")
from pre_tools import load_dict, remove, remove_bt_a, get_text_split_anchors
 
link_prob, commonness, relatedness = load_dict()

train_file = "../train_data/more_links_restrict.txt"
train_output_name = 'enwiki_detect_link_training_data.json'
test_output_name = 'enwiki_detect_link_testing_data.json'

def bi_tri_grams(text_split):
    temp_list = deepcopy(text_split)
    for gram_num in range(2, 4):
        bi_tri_grams = [' '.join(gram).strip() for gram in list(ngrams(temp_list, gram_num))]
        temp_list.extend(bi_tri_grams)
    gram_in_common = [gram for gram in temp_list if gram in commonness and gram.strip() and len(gram.split())<=3]
    return gram_in_common

def location_spread(page_list, search_word):
    word_ngram_num = len(search_word.split())
    if search_word in page_list:
        search_list = page_list
    else:
        search_list = [' '.join(gram) for gram in ngrams(page_list, word_ngram_num)]
    index_list = [idx for idx, token in enumerate(search_list) if token == search_word]
    first_appear = index_list[0]/len(page_list)
    last_appear = index_list[-1]/len(page_list)
    word_spread = last_appear-first_appear
    freq = len(index_list)
    first_index = index_list[0]
    return freq, first_appear, last_appear, word_spread, first_index


def modify_clean_text_split(remove_text, tag_split, linked_list):
    '''replace content without punctuation marks with anchors so that anchor like 'H. N. Abrams' can be found in text split list
    '''
    href_index = []
    for num, piece in enumerate(tag_split):
        if piece.startswith('href'):
            href_index.append(num)

    for idx, anchor in zip(href_index, linked_list):
        if anchor.strip():  # incase some anchors are like ' ' will cover other words cause error
            split_anchor = anchor.split()
            length = len(split_anchor)
            remove_text[idx:idx+length] = split_anchor
        else:
            continue
            
    # deal with special case, can add other cases observed
    temp = ' '.join(remove_text).replace('serine threonine', 'serine/threonine') 
    remove_text = temp.split()
    return remove_text


def get_detect_link_feature(input_article):
    clean_text_split, text_wt_tag, all_anchors, _ = get_text_split_anchors(input_article)
    clean_text_split = modify_clean_text_split(clean_text_split, text_wt_tag, all_anchors)
    orig_word_common = bi_tri_grams(clean_text_split)
    
    temp_features = []
    for word in orig_word_common:
        label = 0
        if word in all_anchors and word.strip():
            label = 1
        frequency, first_occurrence, last_occurrence, spread, _ = location_spread(clean_text_split, word)
        link_probability = link_prob[word]
        temp_features.append([word, link_probability, frequency, first_occurrence, last_occurrence, spread, label])
    return temp_features
    
def save_feature(file, start, end):
    data = []
    count = 1
    for line in file.readlines()[start:end]:
        if count%10==0:
            print(count)
        count+=1
        data.extend(get_detect_link_feature(line))
    df = pd.DataFrame(training_data, columns=column_index)
    df.drop_duplicates()
    df.dropna()
    data_write = df.to_json(orient='split')
    return data_write

def main():
    with open(train_file, 'r') as f:
        column_index = ["anchors", "link_prob", "frequency", "first", "last", "spread", "label"]
        training_data_write = save_feature(f, 0, 105000)
        training_data_write = save_feature(f, 105000, 140000)
    
    with open(train_output_name, 'w') as g:
        g.write(training_data_write)
    
    with open(test_output_name, 'w') as g:
        g.write(testing_data_write)


if __name__ == '__main__':
    main()