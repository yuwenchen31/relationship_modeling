#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#### The following codes are adapted from https://github.com/BoulderDS/LARN with some modification


from LARN_constants import *
from LARN_utils import *
from LARN_modules import * 
from LARN import *

from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import torch
import numpy as np 
import pickle
import pandas as pd
import os
import nltk
import seaborn as sns

nltk.download('vader_lexicon')

model_name = "LARN"
parent_dir = os.path.abspath('..')



# load the original LARN data 
bmap, cmap, wmap, revmap, span_data, span_size, target_pred_ix_set, We = torch.load(parent_dir + "/data/larn_data_smaller.pt")

# Get us-ru
books, chars, all_spans, all_masks, all_months, all_texts = span_data[5]

# ===========================================
# Descriptor generation per topic 

for tpnr in Topicnr: 
    
    print(f"The descriptors for Topic{tpnr} is \n")
    descriptor_log = parent_dir + '/LARN/outputs/descriptors_model_' + model_name + str(tpnr) + '.log'
    model_object_file_name = parent_dir + f"/LARN/outputs/trained-model-LARN-U.S.-Russia-topic{tpnr}.pt"
    model = torch.load(model_object_file_name)

    target_word_ix_set = set()
    # target_word_ix_counter shows the counts of predicates 
    target_word_ix_counter = pd.read_pickle(parent_dir + "/target_word_ix_counter.pk")

    # ge the most common verb/predicate & add into the set 
    for wix, _ in target_word_ix_counter.most_common()[:500]: # can also do without this 500 top word limit
        target_word_ix_set.add(wix)


    log = open(descriptor_log, 'w')


    # shape (30,300)
    R = torch.t(model.l_recon.weight).detach().cpu().numpy()


    # 0,1,2,3...29
    for ind in range(len(R)):

        desc = R[ind] / np.linalg.norm(R[ind])
        top_desc_list = []

        sims = We.dot(np.transpose(desc))

        # sort sims, return the index of minimum value to maximum value
        # [::-1] -> reverse the list -> index of max to min
        # where we get the word index of descriptors
        ordered_words = np.argsort(sims)[::-1]

        desc_list = []
        num_satisfied = 0
        for w in ordered_words:
            if num_satisfied >= num_descs_choice:
                break
            if w in target_word_ix_set:
                desc_list.append(revmap[w])
                num_satisfied += 1


        sentiment_score = calc_desc_sentiment(desc_list)
        top_desc_list.append('-'.join(desc_list) + ' [' + str(sentiment_score) + ']')


        print('descriptor %d:' % ind)
        print(top_desc_list)
        log.write(' '.join(top_desc_list) + '\n')
    print("\n")
    log.flush()
    log.close()

# ===========================================
# Generate topic list 
# topic = pd.read_pickle(parent_dir + "/Topic/info-us-ru.pk")
# topic.pop(-1)
# topic_lists = []
# for t in topic.values():
#     topic_list = [word[0] for word in t]
#     topic_lists.append(topic_list)

# manually filter the topics
topic_lists = [['mueller','counsel','investigation','campaign','robert mueller','election'],
               ['china','canada','india','japan','australia', 'world'],
               ['fbi','james comey','comey','investigation', 'fbi director','campaign','election'],
               ['oil', 'production', 'opec', 'prices', 'saudi arabia', 'crude oil']]

# ===========================================
# Temporal Trend per topic 

# read topic embs
topic_embs = pd.read_pickle(parent_dir + "/Topic/embedding-us-ru.pk")

sample_dict = defaultdict(dict)
desc_dist_dict = dict()

with torch.no_grad():
    
    for tpnr in range(4): 
        
        desc_sample_file_name = 'desc_selected_sample_dict_model_' + model_name + str(tpnr) + '.pkl'
        trajectory_log = parent_dir + '/LARN/outputs/trajectories_model_' + model_name + str(tpnr) + '.log'
        model_object_file_name = parent_dir + f"/LARN/outputs/trained-model-LARN-U.S.-Russia-topic{tpnr}.pt"
        
        model = torch.load(model_object_file_name)
        
        single_topic_emb = topic_embs[tpnr,:]
        c1_name, c2_name = [cmap[c] for c in chars]
        b_name = bmap[books[0]]

        # e.g., US AND Russia
        rel = c1_name + ' AND ' + c2_name


        sample_dict[b_name][c1_name+' AND '+c2_name] = dict()
        for span_index, sample in enumerate(all_texts):
            sample_dict[b_name][c1_name+' AND '+c2_name][span_index] = sample

        char1 = [chars[0]]
        char2 = [chars[1]]

        model.zero_grad()


        masked_spans = []
        noun_spans = []
        for span_index, (span, mask) in enumerate(zip(all_spans, all_masks)):
            masked_span = [span[i] for i in range(len(span)) if mask[i] == predicate_ix or mask[i] == verb_ix]
            masked_spans.append(masked_span)
            noun_span = [span[i] for i in range(len(span)) if mask[i] == noun_ix]
            noun_spans.append(noun_span)

        # outputs_l_rels = da = (5000,30)
        print("Training...")
        _outputs, outputs_l_rels = model(masked_spans, noun_spans, char1, char2, all_months)

        print("Finish training...")

        # save descriptor distribution data
        print("Saving descriptor data...")

        # per country pair, there are (# of examples=5000) of da (shape=30) 
        desc_dist_dict[rel] = dict()
        with open(trajectory_log, 'w') as f: 
            traj_writer = csv.writer(f)
            traj_writer.writerow(['Book', 'Char 1', 'Char 2', 'Span ID'] + ['Topic ' + str(i) for i in range(num_descs)])

            # span_index = 0-4999
            for span_index, olr in enumerate(outputs_l_rels.detach().cpu().numpy()):
                traj_writer.writerow([b_name, c1_name, c2_name, all_months[span_index], span_index] + [o for o in olr])
                desc_dist_dict[rel][span_index] = olr
                
        # VIZ - temporal trend 
        vis_dict = dict()
        num_nouns_choice = 10
        num_top_descs = 3 # to get an overview for all descriptors, change this to 30

        desc_selected_sample_dict = defaultdict(dict) # key: (int_desc_index, int_month_info)
        major_desc_sum_dict = defaultdict(float)

        rmn_traj = read_csv(trajectory_log)
        rmn_descs = read_descriptors(descriptor_log)


        for book in rmn_traj:
            for rel in rmn_traj[book]:


                desc_selected_sample_dict[book][rel] = defaultdict(list)
                plt.close()

                # for each example, the distribution of prob (weights) over 30 descriptors 
                # list of 5000, each has 30 prob
                rtraj = rmn_traj[book][rel]['distributions']

                # list of 5000, each is a month. [15, 21, 15,...]
                mtraj = rmn_traj[book][rel]['months']

                # list of 5000, [0,1,2,...]
                itraj = rmn_traj[book][rel]['span_index']

                data_d = dict()
                desc_sum_d = defaultdict(float)
                trivial_descs = set()

                # find non-trivial descs
                for i in range(num_descs):

                    # {0:defaultdict(list, {}), 1:defaultdict(list, {}),...29:}
                    # len(data_d) = 30
                    data_d[i] = defaultdict(list)

                # for each example
                # r = list of 30 prob
                # m = 15, 21, 23, etc
                for r, m in zip(rtraj, mtraj):

                    # for each distribution in each example: 
                    # i=0,1,2,...29
                    # desc=probs ()

                    for i, desc in enumerate(r):

                        # data_d[i] -> 1st descriptor 
                        #!!! data_d[i][m] -> i-th descriptor's probs at month m (there will be multiple probs in a month)
                        # 
                        data_d[i][m].append(desc)

                # data_d.keys = 0,1,2,..29
                for i in data_d.keys():
                    trivial = True
                    # data_d[i].keys() = [15, 21,24, 22,..] where i-th descriptor appears and has probs
                    for m in data_d[i].keys():

                        # for i-th desc, the prob is sum(each month's mean prob)
                        desc_sum_d[i] += np.mean(data_d[i][m])

                # sorted(.., key:...)[:num_top_descs] -> sort according to prob, then slice to top 3 
                #[(0, nan),
                # (20, 2.571797799319029),
                # (5, 2.448546141386032),
                # (19, 2.1164665557444096)]
                # top_ds = the desc index which have highest prob
                top_ds = [top_d[0] for top_d in sorted([(i, share) for i, share in desc_sum_d.items()],
                                               key=lambda x: -x[1])]#[:num_top_descs]]
                
                
                # NEW: try to get the top pos/neg/neutral desc
                rmn_descs_list = list(rmn_descs.items())
                ordered_descs = [rmn_descs_list[i] for i in top_ds]

                neg = []
                pos =[] 
                ntl = []
                for des in ordered_descs:
                    if '0.0' not in des[1]:
                        if '-0.' in des[1]:
                            neg.append(des)
                        else: 
                            pos.append(des)
                    else: 
                        ntl.append(des)

                top_ds_pnn = [pos[0][0], neg[0][0], ntl[0][0]]
                

                # print the descriptors and sentiment score 
                print(rmn_descs[top_ds_pnn[0]], rmn_descs[top_ds_pnn[1]], rmn_descs[top_ds_pnn[2]])

                seaborn_d = {'month_info': [], 'desc_share': [], 'desc_type': []}
                desc_share_dict = defaultdict(list)

                # go in each example:
                # r=list of 30 probs
                # m= e.g., 15
                # span_index=e.g.,0
                for r, m, span_index in zip(rtraj, mtraj, itraj):

                    # for each 30 desc prob 
                    for i, desc in enumerate(r):

                        # we only go to top 3 desc
                        if i not in top_ds_pnn:
                            continue

                        # (i,m) = e.g.,(5,15)= 5th desc at 15th month, (5, 24)=5th desc at 24th month
                        desc_share_dict[(i, m)].append((desc, span_index))
                        seaborn_d['month_info'].append('20' + month_to_str(m, year_base))
                        seaborn_d['desc_share'].append(desc)
                        seaborn_d['desc_type'].append(i)
                        major_desc_sum_dict[i] += desc



                for k in desc_share_dict.keys():
                    desc_share_dict[k].sort(key=lambda x: -x[0])
                    if len(desc_share_dict[k]) < sample_sel_num:
                        continue
                    for sel_i in range(sample_sel_num):
                        desc_selected_sample_dict[book][rel][k].append(sample_dict[book][rel][desc_share_dict[k][sel_i][1]])

                vis_dict[rel] = [seaborn_d]

        pickle.dump(desc_selected_sample_dict, open(desc_sample_file_name, 'wb'))


        # Visualization 
        print('##################', rel, '##################')
        sns.set_style("ticks")
        sns.set_context("notebook", font_scale=3.2, rc={"lines.linewidth": 2}) # previous font: 2.2

        import re 

        desc_share_df = pd.DataFrame(data=vis_dict[rel][0]).sort_values(by=["month_info"])
        # normalize the weights - min-max normalization
        desc_share_df['desc_share'] = (desc_share_df['desc_share']-desc_share_df['desc_share'].min())/(desc_share_df['desc_share'].max()-desc_share_df['desc_share'].min())
        plt.figure(figsize=(30,10))


        # make the color coding systematic - min sentiment score: red, medium: skyblue, max: green
        top_3_desc_nr = set(vis_dict[rel][0]['desc_type'])
        sentiment_score = dict()

        for i in top_3_desc_nr:
            sentiment_score[i] = float(re.search('(?<=\[).+?(?=\])', rmn_descs[i]).group())
        sentiment_score = dict(sorted(sentiment_score.items(), key=lambda item: item[1]))
        colors = ['red', 'skyblue', 'green']
        my_pal = dict()
        for k,c in zip(sentiment_score.keys(),colors):
            my_pal[k] = c

        ax = sns.pointplot(data=desc_share_df, x='month_info', y='desc_share', hue='desc_type',
                           ci=None, palette=my_pal, markers=['o', 'x', '^'], scale=1.5)

        plt.setp([ax.get_lines()],alpha=.5)
        ax.legend(loc='center left', bbox_to_anchor=(0, 1.3))

        ax.legend_.set_title(f"Topic: {','.join(topic_lists[tpnr])}")

        for t in ax.legend_.texts:
            t.set_text(rmn_descs[int(t.get_text())].split(','))

        key_event_count = 0

        for xl in ax.get_xticklabels():
            if xl.get_text()[2:] in key_event_dict['Internation'][rel].keys():
                key_event_text = key_event_dict['Internation'][rel][xl.get_text()[2:]].split('(')[0]
                x_coor = str_to_month(xl.get_text()[2:], year_base) - 1
                plt.axvline(x=x_coor, linestyle='--', color='black', alpha=0.5, lw = 2)
                key_event_count += 1
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
        ax.xaxis.set_label_text('')
        ax.yaxis.set_label_text('Descriptor Weight')
        #plt.show()
        plt.savefig(plot_save_path + f'LARN-TempTrend-{cmap[chars[0]]}-{cmap[chars[1]]}-{tpnr}.png', bbox_inches='tight')

        desc_share_df.to_pickle(model_save_path + f"LARN-TimeSeries-{cmap[chars[0]]}-{cmap[chars[1]]}-{tpnr}.pk")

        # TODO: Create sentiment score plot for each topic 







