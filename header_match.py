import pandas as pd
from fuzzywuzzy import fuzz, process
import string
import pickle as pkl
import numpy as np
import time
import re
from multiprocessing import Pool

# Global constants, should be in input_Params
THRESHOLD = 85

# can be in input_params, but this is stuff that needs to be replaced or removed.
replace_char = ["*", "%", "/", "\\"]
strip_char = ["'", "-", ".", "!", ":", ";"]
num_char =  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
red_flag_char = [" AVE ", " ST ", " AV ", " BLVD ", " BLV ", " RD ", " DR "]
common_errors = {
    "0": "O",
    "1": "l",
    "5": "S",
    "8": "B"
}

def clean_header(h_raw):
    # cleans the header
    red_flag = False
    h = h_raw.upper()
    h = h.partition(' (')[0].partition('-CONT')[0].partition('—CONT')[0]
    for s in replace_char:
        h = h.replace(s, "")
    for s in strip_char:
        h = h.strip(s)
    cnt = 0
    hl = []
    for c in list(h):
        if c in common_errors: c = common_errors[c]
        if c in num_char: cnt += 1
        hl.append(c)
    h = ''.join(hl).upper()
    for rf in red_flag_char:
        if rf in h: red_flag = True
    if cnt > 3 or red_flag: 
        h = ""
    return h.upper()
    
def assign_clean(D):
    # assigns to dataframe
    return [clean_header(h) for h in D["Header"]]

def header_scorer(string1, string2):
    # Scores the fuzzy match between the ocr header and the true header
    return (fuzz.partial_ratio(string1,string2) + fuzz.ratio(string1,string2))/2.0

def match_header(input_tuple):
    # Matches ocr headers to true headers
    header, true_headers = input_tuple
    results = process.extractOne(header, true_headers, scorer=header_scorer)
    return (header,results[0],results[1])


# driver function to create the map_dict
def generate_dict(df, true_headers, raw_dict):

    map_dict = raw_dict
    t1 = time.time()
    header_set = set(clean_header(h) for h in df['Header'].tolist() if (len(h) < 150) and (len(h) > 2) and (h != ""))
    header_set = header_set - map_dict.keys()

    t2 = time.time()

    print(str(len(header_set)) + ' headers to match')
    pool = Pool(4)
    input_list = [(header, true_headers) for header in header_set]
    results_list = pool.map(match_header,input_list)
    for results in results_list:
        header,header_match,score = (results[0],results[1],results[2])
        #print(header,header_match,score)
        map_dict[header] = (header_match,score)
    #map_dict = match(list(header_set), true_headers, map_dict)
    t3 = time.time()
    print('Done in : ' + str(round(t3-t2,3)))
    pkl.dump(map_dict, open('header_match_dict.pkl', 'wb'))

    return map_dict

# driver function to header match given a map_dict
def match_headers(df, map_dict):

    #df = df.drop_duplicates("Header").dropna().assign(clean_headers=assign_clean)
    t1 = time.time()

    header_dict = {}
    score_dict = {}
    bool_dict = {}

    for header in set(df['Header']):
        cleaned_header = clean_header(header)
        if cleaned_header in map_dict.keys():
            if map_dict[cleaned_header][1] >= 85.0:
                header_dict[header] = map_dict[cleaned_header][0]
                score_dict[header] = map_dict[cleaned_header][1]
                bool_dict[header] = True
            else:
                header_dict[header] = cleaned_header
                score_dict[header] = map_dict[cleaned_header][1]
                bool_dict[header] = False
        else:
            header_dict[header] = cleaned_header
            score_dict[header] = 0
            bool_dict[header] = "ERR: NOT IN MAPDICT"

    t2 = time.time()
    print('clean header assigning time: ' + str(round(t2-t1,3)) + ' s')

    t1 = time.time()
    header_list = []
    score_list = []
    bool_list = []

    for row in df.itertuples():
        raw_header = row.Header
        header_list.append(header_dict[raw_header])
        score_list.append(score_dict[raw_header])
        bool_list.append(bool_dict[raw_header])
    
    t2 = time.time()
    print('assigning time: ' + str(round(t2-t1, 3)) + ' s')
    df = df.assign(clean_header=header_list,score=score_list,matched=bool_list)

    return df





