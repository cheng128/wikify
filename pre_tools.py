import re
import json
import joblib
from nltk import ngrams
from copy import deepcopy
from bs4 import BeautifulSoup
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_dict():
    dict_link_prbability = json.load(open(os.path.join(BASE_DIR,'data_model/link_prob_dict.json')))
    dict_commonness = json.load(open(os.path.join(BASE_DIR, 'data_model/commonness_dict.json')))
    dict_relatedness = json.load(open(os.path.join(BASE_DIR,'data_model/relatedness_dict.json')))
    return dict_link_prbability, dict_commonness, dict_relatedness

def load_model():
    disambi_model = joblib.load(os.path.join(BASE_DIR,'data_model/disambiguate_trained_model'))
    detect_model = joblib.load(os.path.join(BASE_DIR,'data_model/detect_link_with_trained_model'))
    return disambi_model, detect_model

def remove(documents):
    remove_chars = '[!()*+,./:;<=>?@，。、…【】《》^_`{|}~"]+'
    return re.sub(remove_chars, ' ', documents)

def remove_bt_a(doc):
    remove_list = '/a>[!()*+,./:;<=>?@·，。、…【】《》^_`{|}~–-]?'
    doc = doc.replace('a><a', 'a> <a')
    doc = doc.replace('\xa0', ' ')
    return re.sub(remove_list, '/a> ', doc)

def get_text_split_anchors(doc):
    '''inpute: orig content with tag
    output: content without tag, content split (['href magic%20bean magic' ,bean a]), all_anchors
    '''
    text_wt_tag = json.loads(doc)["text"]
    # deal with /a>.<a problem, or like /a>-link<a, won't remove punctuation marks of anchors(between <a> tag)
    text_wt_tag =  remove_bt_a(text_wt_tag)
    soup = BeautifulSoup(text_wt_tag, 'lxml')
    anchors = [tag.text for tag in soup.find_all('a')]
    hrefs = [tag['href'] for tag in soup.find_all('a')]
    text_split = soup.text.split()
    # content splited and remove -,. .etc punctuation marks
    remove_text_split = [remove(x).strip() for x in text_split] 
    # use to find the location and spread of anchors(if start with 'href' then it is an anchor)
    wt_tag_split = [a for a in text_wt_tag.split() if '<a' not in a and a] 
    return remove_text_split, wt_tag_split, anchors, hrefs