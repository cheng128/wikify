import re
from nltk import ngrams
from disambi_detect import gen_need_disambi, word_disambiguation, bi_tri_grams

def remove(documents):
    remove_chars = '[!()*+,./:;<=>?@，。、…【】《》^_`{|}~"-]+'
    return re.sub(remove_chars, ' ', documents)

def get_return_info(split_text, ngram_list):  
    words_href, words_index, need_dis = gen_need_disambi(split_text, ngram_list) 
    final_anchor = word_disambiguation(words_href, need_dis)
    return_list = []
    for anchor in final_anchor.keys():
        temp = {}
        temp['entity'] = anchor
        temp['url'] = 'https://en.wikipedia.org/wiki/' + words_href[anchor]
        temp['start'] = words_index[anchor]
        temp['end'] = temp['start']+len(anchor.split())-1
        return_list.append(temp)
    return return_list

def correct_clean_idx(target_text_split, temp_info):
    transfer_idx = [idx for idx, token in enumerate(target_text_split)
                       for num in range(len(remove(token).split()))]
    for info in temp_info:
        temp_start = info['start']
        temp_end = info['end']
        info['start'] = transfer_idx[temp_start]
        info['end'] = transfer_idx[temp_end]
    return temp_info

def disambiguate_detect(text):
    orig_text_split = text.split()
    clean_text_split = remove(text).split()
    orig_ngrams = bi_tri_grams(orig_text_split)
    clean_ngrams = bi_tri_grams(clean_text_split)
    orig_info = get_return_info(orig_text_split, orig_ngrams)
    orig_start = [idx for info in orig_info for idx in range(info['start'], info['end']+1)]
    clean_temp_info = get_return_info(clean_text_split, clean_ngrams)
    clean_info = correct_clean_idx(orig_text_split, clean_temp_info)
    orig_info.extend(clean_info)
    return orig_text_split, orig_info

def chin_name_idx(text):
    all_name = re.findall('[A-Z]+[a-z]+ [A-Z]+[a-z]+-+[a-z]+', text)
    all_name.extend(re.findall('[A-Z]+[a-z]+ [A-Z]+[a-z]+ [A-Z]+[a-z]+', text))
    all_name.extend(re.findall('[A-Z]+[a-z]+-[a-z]+', text))
    all_name.extend(re.findall('[A-Z]+[a-z]+ [A-Z]+[a-z]+', text))
    remove_chars = '[!()*+,./:;<=>?@，。、…【】《》^_`{|}~"]+'
    text = re.sub(remove_chars, ' ', text)
    idx_list = []
    for name in all_name:
        ngram_list = [' '.join(gram) for gram in ngrams(text.split(), len(name.split()))]
        idx_list.extend([idx+i for idx, gram in enumerate(ngram_list) for i in range(len(gram.split())) 
                    if gram in all_name])
    return all_name, idx_list

def remove_duplicate_result(text, result):
    name_list, name_idx = chin_name_idx(text)
    sort_length = sorted(result, key=lambda x: len(x['entity'].split()), reverse=True)
    sort_idx = sorted(sort_length, key=lambda x: x['start'])
    exist_word, exist_idx = [], []
    for info in sort_idx:
        if info['entity'] not in exist_word and info['start'] not in exist_idx:
            exist_word.append(info['entity'])
            exist_idx.extend([i for i in range(info['start'], info['end']+1)])
        else:
            result.remove(info)
    final_result = []
    for info in result:
        if info['start'] not in name_idx or info['entity'] in name_list:
            final_result.append(info)
        else:
            pass
    final_result = sorted(final_result, key=lambda x: x['start'])
    return final_result
    
def detect_disambi_result(text):
    orig_text_split, temp_result = disambiguate_detect(text)
    final_result = remove_duplicate_result(text, temp_result)
    return final_result