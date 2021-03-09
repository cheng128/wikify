import re
import pandas as pd
from math import log
from nltk import ngrams
from copy import deepcopy
from pre_tools import load_dict, load_model

link_prob, commonness, relatedness = load_dict()
disambi_model, detect_model = load_model()

def bi_tri_grams(text_split):
    temp_list = deepcopy(text_split)
    for gram_num in range(2, 4):
        bi_tri_grams = [' '.join(gram).strip() for gram in list(ngrams(temp_list, gram_num))]
        temp_list.extend(bi_tri_grams)
    gram_in_common = [gram for gram in temp_list if gram in commonness and gram.strip()]
    return gram_in_common

def location_spread(page_list, search_word):
    word_ngram_num = len(search_word.split())
    if search_word in page_list:
        search_list = page_list
    else:
        search_list = [' '.join(gram) for gram in ngrams(page_list, word_ngram_num)]
    index_list = [idx for idx, token in enumerate(search_list) if token == search_word]
    first_appear = round(index_list[0]/len(page_list), 3)
    last_appear = round(index_list[-1]/len(page_list), 3)
    word_spread = round(last_appear-first_appear,3)
    freq = len(index_list)
    first_index = index_list[0]
    return freq, first_appear, last_appear, word_spread, first_index

def cal_relat(A, B):
    """ Utilize two href to calculate relatedness between two words.
    formula: relatedness(a, b) = (log(max(|A|,|B|))-log(|A&B|))/log(W)-log(min(|A|, |B|))
    Where a and b are the two articles of interest, A and B are the sets of all articles that link to a and b respectively, and W is set of all articles in Wikipedia.
    """
    all_article_num = 61784326
    log_W = log(all_article_num)
    link_article = [relatedness[href] for href in [A, B]]
    article_count = [len(article) for article in link_article]
    intersection_count = len(set(link_article[0])&set(link_article[1]))
    intersection_log = 0
    if intersection_count!=0:
        intersection_log = log(intersection_count)
    return ((log(max(article_count))-intersection_log) / (log_W-log(min(article_count))))

def detect_link_predict(target_list, words_list):
    """ Get features for detect link(link probability, frequency, first occurrence, last occurrence, spread) and predict whether a word need to be linked.
    """
    detect_data = []
    first_index_list = []
    for target in words_list:
        probability = link_prob[target]
        frequency, first_occurrence, last_occurrence, spread, first_index = location_spread(target_list, target)
        first_index_list.append(first_index)
        detect_data.append([probability, frequency, first_occurrence, last_occurrence, spread])
    dataframe = pd.DataFrame(detect_data, columns=["link prob", "frequency", "first", "last", "spread"])
    final_link_result = detect_model.predict(dataframe)
    return final_link_result, first_index_list

def gen_need_disambi(text_split, text_ngrams):
    link_result, first_place = detect_link_predict(text_split, text_ngrams)
    link_word_idx = [idx for idx, label in enumerate(link_result) if label==1]
    word_idx_dict = {text_ngrams[index]: first_place[index] for index in link_word_idx}
    temp_no_need = [gram for gram in word_idx_dict if len(commonness[gram])==1]
    temp_no_need_href = [commonness[no_need][0][0] for no_need in temp_no_need]
    temp_need = [word for word in word_idx_dict if word not in temp_no_need]
    
    if not temp_no_need:
        # {highest_commonness: [highest_commonness_href, word]}
        common_href_word = {commonness[word][0][1]: [commonness[word][0][0], word] for word in word_idx_dict}
        if common_href_word:
            max_common = max(common_href_word.keys()) 
            temp_no_need_href.append(common_href_word[max_common][0])
            temp_no_need.append(common_href_word[max_common][1])
            temp_need.remove(common_href_word[max_common][1])
        else:
            all_commonness = [(commonness[token][0][1], token, commonness[token][0][0], idx) 
                              for token, idx in zip(text_ngrams, first_place)]
            max_common_token = max(all_commonness, key= lambda x:x[0])
            temp_no_need_href.append(max_common_token[2])
            temp_no_need.append(max_common_token[1])
            word_idx_dict[max_common_token[1]] = max_common_token[3]
    word_href = {word: href for word, href in zip(temp_no_need, temp_no_need_href)}
    return word_href, word_idx_dict, temp_need


def word_disambiguation(no_need_href, need_list):
    """ This function will use words that don't need to do disambiguation to calculate relatedness and then use context quality, commonness of that sense to disambiguate others.
    """
    anchor_href = {}
    for ambi in need_list:
        ambi_data = []
        possible_page = commonness[ambi]
        sense_url = [url_com[0] for url_com in possible_page]
        for page in possible_page:
            sense_commonness = page[1]/100
            temp_relatedness = [cal_relat(page[0],no_dis) for no_dis in no_need_href.values()]
            temp_weight = [(cal_relat(page[0], h)+link_prob[w])/2 for w, h in no_need_href.items()]
            weight_relate = [w*r for w, r in zip(temp_weight, temp_relatedness)]
            sense_relatedness = sum(weight_relate)/sum(temp_weight)
            text_quality = sum(temp_weight)
            ambi_data.append([sense_commonness, sense_relatedness, text_quality])

        dataframe = pd.DataFrame(ambi_data,
                                 columns=['commonness', 'relatedness', 'context_quality'])
        
        sense_probability = disambi_model.predict_proba(dataframe)
        sense_label_1 = [pair[1] for pair in sense_probability]
        result = max(zip(sense_url, sense_label_1), key = lambda x: x[1])
        anchor_href[ambi] = result[0]
    no_need_href.update(anchor_href)
    return no_need_href