#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:26:46 2022

@author: yuwenchen
"""

# filter articles for LARN 

import pandas as pd
import torch
import pickle
import os 

# parent directory 
p = os.path.abspath('..')

_, _, _, _, span_data, _, _, _ = torch.load(p + '/data/larn_data_smaller.pt')
topic_info = pd.read_pickle(p + "/ARM/Topic/info-us-ru.pk")

# filter out the topic words 
topic_lists = [['mueller','counsel','investigation','campaign','robert mueller','election'],
               ['china','canada','india','japan','australia', 'world'],
               ['fbi','james comey','comey','investigation', 'fbi director','campaign','election'],
               ['oil', 'production', 'opec', 'prices', 'saudi arabia', 'crude oil']]


# get data from us-ru 
us_ru_topic = span_data[5]
book, chars, spans, masks, months, texts = us_ru_topic 

# use keywords to filter the article 
topic_span = []
topic_mask = []
topic_month = []
for i, topic in enumerate(topic_lists):
  span_list = []
  mask_list = []
  month_list = []
  for span, mask, month, text in zip(spans, masks, months, texts):
    if any(x in text[0] for x in topic_lists[i]):
      span_list.append(span)
      mask_list.append(mask)
      month_list.append(month)
  topic_span.append(span_list)
  topic_mask.append(mask_list)
  topic_month.append(month_list)
  
  
# check number of articles
for a,b,c in zip(topic_span, topic_mask, topic_month):
  print(len(a))
  print(len(b))
  print(len(c))
  
data = book, chars, topic_span, topic_mask, topic_month

# save the texts for us-ru
with open(p + '/LARN/LARN-us-ru-4tp.pk', 'wb') as f:
    pickle.dump(data, f)