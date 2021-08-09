import sys
import json
import pandas as pd
from math import log
from bs4 import BeautifulSoup
from pre_tools import load_dict
sys.path.append('..')

link_prob, commonness, relatedness = load_dict()

train_file = "../train_data/anchors_btwn_10_20_all.txt"
output_file_name = 'enwiki_disambiguation_training_data.json'

def cal_relate(A, B):
    """ Utilize two href to calculate relatedness between two words.
    formula: relatedness(a, b) = (log(max(|A|,|B|))-log(|A&B|))/log(W)-log(min(|A|, |B|))
    Where a and b are the two articles of interest, A and B are the sets of all articles 
    that link to a and b respectively, and W is set of all articles in Wikipedia.
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

def get_no_need_dis(json_text):
    return_data = []
    line_json = json.loads(json_text)
    text_id = line_json["id"]
    text_content = line_json["text"]
    text_content_soup = BeautifulSoup(text_content, 'lxml')
    anchors_href = [(a.text, a['href']) for a in text_content_soup.find_all('a') 
                                        if len(a.text.split())<=3 and a.text.strip()]
    no_need = [anhr for anhr in anchors_href if len(commonness[anhr[0]])==1]
    need = [anhr for anhr in anchors_href if anhr not in no_need]
    
    if not no_need:
        common_href_word = {commonness[word[0]][0][1]: [commonness[word[0][0]], word]
                            for word in need}
        max_common = max(common_href_word.keys()) 
        no_need = [common_href_word[max_common][1]]
        need.remove(common_href_word[max_common][1])
    return no_need, need

def get_features(pages, no_need, need, target_ac, target_href):
    temp = []
    for page in pages:
        label = 0
        sense_commonness = page[1]/100
        temp_relatedness = [cal_relate(page[0], no_dis[1])  for no_dis in no_need]
        temp_weight = [(cal_relate(page[0], anchor[1])+link_prob[anchor[0]])/2
                       for anchor in no_need]
        weight_relate = [w*r for w, r in zip(temp_weight, temp_relatedness)]
        sense_relatedness = sum(weight_relate)/sum(temp_weight)
        text_quality = sum(temp_weight)
        if target_href == page[0]: label = 1
        temp.append([sense_commonness, sense_relatedness, text_quality, label])   
    return temp
    
def get_train_feature(json_text):
    no_need_dis, need_dis = get_no_need_dis(json_text)
    return_data = []
    for ac, hr in need_dis:
        possible_page_common = commonness[ac]
        return_data.extend(get_features(possible_page_common, no_need_dis, need_dis, ac,hr))
    return return_data

def main():
    with open(train_file, 'r') as f:
        features = []
        count = 1
        for line in f.readlines():
            if count%100==0:
                print(count)
            count += 1
            features.extend(get_train_feature(line))
            
        column_index = ["commonness", "relatedness", "context_quality", "label"]
        df = pd.DataFrame(features, columns=column_index)
        df.drop_duplicates()
        df.dropna()
        training_data_write = df.to_json(orient='split')
        
    with open(output_file_name, 'w') as g:
        g.write(training_data_write)


if __name__ == "__main__":
    main()