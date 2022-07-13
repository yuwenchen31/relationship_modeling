from constants import *

import pickle
import re 
import numpy as np
import pandas as pd 
import csv
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import seaborn as sns
import torch.nn as nn
import torch
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
import spacy
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

np.random.seed(2022)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def preprocess_bertopic(docs, entity_list): # TODO: remove entity? 
    
    """
    Remove stopwords & entity. Convert each document into string. 
    
    """
    #print("Loading language model...")
    #spacy.require_gpu()
    #nlp_model = spacy.load("en_core_web_sm")
    stop_words = stopwords.words('english')
    filter_set = set(stop_words + entity_list)
    filtered_doc = []
    n = 0
    print("Starting preprocessing!")
    #docs = [d[0] for d in docs]
    for doc in docs:
        if n%10000 ==0: 
            print(f"We are at {n} documents!")
        #output = nlp_model(doc)
        #filtered = ' '.join([token.lemma_ for token in output if token.lemma_ not in entity_list and token.lemma_ not in stop_words])
      
        tokenize = word_tokenize(doc[0])
        filtered = ' '.join([word for word in tokenize if word not in filter_set])
        filtered_doc.append(filtered)
        n += 1
        
    return filtered_doc


def scaled_dot_product_attention(q,k,v,mask=None): 
    
    q = q.to(torch.double)
    k = k.to(torch.double)
    v = v.to(torch.double)
    
    
    matmul_qk = torch.matmul(q, torch.t(k))
    
    dk = k.shape[0]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += mask * -1.0e9

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    
    output = torch.matmul(attention_weights, v) 
    
    return output #,attention_weights




def generate_negative_samples(num_traj, span_size, negs, span_data):
    
    # (start, stop, amount) -> rray([14, 38,  6,  5, 32,  0, 29, 43, 55, 57, 44, 64, 19,  9, 35])
    # randomly select entity pairs to generate examples from
    ##inds = np.random.randint(0, num_traj, negs)
    inds = torch.randint(0,num_traj,(negs,), device=device)
    
    # neg_word.shape=(15,2102) -> 15 articles, each is of 2102 words
    ##neg_words = np.zeros((negs, span_size)).astype('int32')
    ##neg_masks = np.zeros((negs, span_size)).astype('float32')
    
    neg_words = torch.zeros((negs,span_size), device=device, dtype=torch.int32)
    neg_masks = torch.zeros((negs,span_size), device=device, dtype=torch.float32)
    
    
    # go through each entity pairs in inds 
    # index = 0,1,2,...,14
    for index, i in enumerate(inds):
        
        ##rand_ind = np.random.randint(0, len(span_data[i][2]))
        rand_ind = torch.randint(0,len(span_data[i][2]),(1,), device=device)
        
        # neg_words[0] = a random article from enity pair 14
        # neg_words[1] = a random article from enity pair 38
        
        neg_words[index] = torch.as_tensor(span_data[i][2][rand_ind], device=device)
        
        # neg_masks[0] = a random mask from entity pair 14
        # neg_masks[1] = a random mask from entity pair 32
        neg_masks[index] = torch.as_tensor(span_data[i][3][rand_ind],device=device)
    return neg_words, neg_masks


# parse learned descriptors into a dict

def read_descriptors(desc_file):
    desc_map = {}
    f = open(desc_file, 'r')
    for i, line in enumerate(f):
        line = line.split()
        desc_map[i] = ', '.join(line)
    return desc_map

# read learned trajectories file, adapted from Mohit's RMN code

def read_csv(csv_file):
    reader = csv.reader(open(csv_file, 'r'))
    all_traj = {}
    prev_book = None
    prev_c1 = None
    prev_c2 = None
    total_traj = 0
    for index, row in enumerate(reader):
        if index == 0:
            continue
        book, c1, c2, month, span_index = row[:5]
        
        
        if prev_book != book or prev_c1 != c1 or prev_c2 != c2:
            prev_book = book
            prev_c1 = c1
            prev_c2 = c2
            if book not in all_traj:
                all_traj[book] = {}
                
            all_traj[book][c1+' AND '+c2] = {'distributions': [], 'months': [], 'span_index': []}
            
            total_traj += 1
            
        all_traj[book][c1+' AND '+c2]['distributions'].append(np.array(row[5:], dtype='float32'))
        
        all_traj[book][c1+' AND '+c2]['months'].append(int(month))
       
        all_traj[book][c1+' AND '+c2]['span_index'].append(int(span_index))
       
    return all_traj


def month_to_str(month, year_base):
    month -= 1
    if month % 12 + 1 < 10:
        str_month = str(year_base + int(month / 12)) + '-0' + str(month % 12 + 1)
    else:
        str_month = str(year_base + int(month / 12)) + '-' + str(month % 12 + 1)
    return str_month


def str_to_month(str_month, year_base):
    str_month_split = str_month.split('-')
    year = int(str_month_split[0])
    month = int(str_month_split[1])
    return (year - year_base) * 12 + month


def desc_query(desc_sample_file_name, rel, desc_i, month):
    desc_selected_sample_dict = pickle.load(open(desc_sample_file_name, 'rb'))
    month_i = str_to_month(month, year_base)
    for doc in desc_selected_sample_dict['Internation'][rel][(desc_i, month_i)]:
        for doc_sent in doc:
            print('\n\n', doc_sent, '\n\n')


def attn_query(attn_sample_file_name, rel, desc_i, month, word):
    attn_selected_sample_dict = pickle.load(open(attn_sample_file_name, 'rb'))
    month_i = str_to_month(month, year_base)
    for doc in attn_selected_sample_dict['Internation'][rel][(desc_i, month_i)][word]:
        for doc_sent in doc:
            print('\n\n', doc_sent, '\n\n')
       

def calc_desc_sentiment(desc_list):
    sia = SentimentIntensityAnalyzer()
    vader_sentiment_result = sia.polarity_scores(' '.join(desc_list))
    return vader_sentiment_result['compound']

# NEW 

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
