1#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# The following codes are adapted from: https://github.com/BoulderDS/LARN/blob/master/our_model.ipynb

from LARN_origin_constants import *
from LARN_origin_utils import *
from LARN_origin_modules import *

import random
import time
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import torch 
import pickle 
import numpy as np 
import torch.nn as nn 

random.seed(2019)

model_id = 'larn'

bmap, cmap, wmap, revmap, span_data, span_size, target_pred_ix_set, We = torch.load(data_load_path)

num_chars, num_books, num_traj = len(cmap), len(bmap), len(span_data)

descriptor_log = 'descriptors_model_' + str(model_id) + '.log'

trajectory_log = 'trajectories_model_' + str(model_id) + '.log'

desc_sample_file_name = 'desc_selected_sample_dict_model_' + str(model_id) + '.pkl'

attn_sample_file_name = 'attn_selected_sample_dict_model_' + str(model_id) + '.pkl'

training_progress_log = 'training_progress_log_' + str(model_id) + '.txt'

model_object_file_name = 'trained_model_' + str(model_id) + '.pt'

A_dict_file_name = 'A_dict_' + str(model_id) + '.pkl'


# ===================================================================================================
# Model 
class LARN(nn.Module):

    def __init__(self, d_word, d_noun_hidden, d_ent, d_meta, d_mix, d_desc, n_ent, n_meta, We):
        super(LARN, self).__init__()
        self.d_desc = d_desc
        tensor_We = torch.FloatTensor(We).to(device)
        self.l_st = TrainedWordEmbeddingLayer(tensor_We, d_word)
        self.l_noun = NounAttentionLayer_SingleQuery(tensor_We, d_word, d_noun_hidden, d_desc)
        self.l_ent = nn.Embedding(n_ent, d_ent)
        self.l_mix = MixingLayer_Attention_SingleQuery_Concat(d_word, d_noun_hidden, d_ent, d_mix)
        self.l_rels = DistributionLayer(d_mix, d_desc)
        self.l_recon = nn.Linear(d_desc, d_word)
        nn.init.xavier_uniform_(self.l_recon.weight)
        
    def init_attention(self, A):
        self.l_noun.query.weight = nn.Parameter(A)

    def forward(self, spans, noun_spans, e1, e2, months):
        e1_tensor = torch.LongTensor(e1).to(device)
        e2_tensor = torch.LongTensor(e2).to(device)
        
        outputs_l_st = self.l_st(spans)
        outputs_l_noun = self.l_noun(noun_spans, months)
        outputs_l_e1 = self.l_ent(e1_tensor).expand(len(spans), -1)
        outputs_l_e2 = self.l_ent(e2_tensor).expand(len(spans), -1)
        
        outputs_l_mix = self.l_mix(outputs_l_st, outputs_l_noun, outputs_l_e1, outputs_l_e2)
        
        outputs_l_rels = self.l_rels(outputs_l_mix)
        
        outputs = self.l_recon(outputs_l_rels)
        return outputs, outputs_l_rels
    
    
if __name__ == "__main__":

    model = LARN(d_word, d_noun_hidden, d_char, d_book, d_mix, num_descs, num_chars, num_books, We).to(device)
    loss_function = Contrastive_Max_Margin_Loss().to(device)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)
    label_generation_layer = TrainedWordEmbeddingLayer(torch.FloatTensor(We), d_word).to(device)
    
    A_dict = dict()
    avg_losses =[]
    
    # TODO: clean up
    book, chars, big_spans, big_masks, big_months, _ = span_data[5]
    
    for epoch in range(n_epochs):
        epoch_start_t = time.time()
        epoch_total_loss = torch.tensor(0.).to(device)
        random.shuffle(span_data)
        
        # TODO: uncomment this when we want to compute all pairs 
        #for book, chars, big_spans, big_masks, big_months, _ in span_data:
        char1 = [chars[0]]
        char2 = [chars[1]]
        
        zip_list_to_shuffle = list(zip(big_spans, big_masks, big_months))
        random.shuffle(zip_list_to_shuffle)
        shuffled_big_spans, shuffled_big_masks, shuffled_big_months = zip(*zip_list_to_shuffle)
        
        split_indices = [i for i in range(0, len(shuffled_big_spans), batch_size)] # split to batches to fit in memory
        spans_split = np.split(shuffled_big_spans, split_indices)
        masks_split = np.split(shuffled_big_masks, split_indices)
        months_split = np.split(shuffled_big_months, split_indices)
        
        for k, (spans, masks, months) in enumerate(zip(spans_split, masks_split, months_split)):
            if len(spans) != batch_size:
                continue
        
            model.zero_grad()

            if (chars[0], chars[1]) in A_dict:
                model.init_attention(A_dict[(chars[0], chars[1])].to(device))
            else:
                nn.init.xavier_uniform_(model.l_noun.query.weight)

            train_masked_spans = []
            noun_spans = []
            drop_masks = (np.random.rand(*(masks.shape)) < (1 - p_drop)).astype('float32')
            for span_index, (span, mask, drop_mask) in enumerate(zip(spans, masks, drop_masks)):
                train_masked_span = [span[i] for i in range(len(span))
                                     if mask[i] == predicate_ix and drop_mask[i] == 1]
                train_masked_spans.append(train_masked_span)
                noun_span = [span[i] for i in range(len(span)) if mask[i] == noun_ix]
                noun_spans.append(noun_span)

            pos_masked_spans = []
            for span_index, (span, mask) in enumerate(zip(spans, masks)):
                pos_masked_span = [span[i] for i in range(len(span)) if mask[i] == predicate_ix]
                pos_masked_spans.append(pos_masked_span)

            neg_spans, neg_masks = generate_negative_samples(num_traj, span_size,
                                                             num_negs, span_data)
            neg_masked_spans = []
            for span_index, (span, mask) in enumerate(zip(neg_spans, neg_masks)):
                neg_masked_span = [span[i] for i in range(len(span)) if mask[i] == predicate_ix]
                neg_masked_spans.append(neg_masked_span)

            outputs, _outputs_l_rels = model(train_masked_spans, noun_spans, char1, char2, months)
            pos_labels = label_generation_layer(pos_masked_spans)
            neg_labels = label_generation_layer(neg_masked_spans)

            R = torch.t(model.l_recon.weight)

            loss = loss_function(outputs, pos_labels, neg_labels, len(spans), R, eps, d_word)

            loss.backward()
            optimizer.step()
            epoch_total_loss += loss

            A = model.l_noun.query.weight.detach().cpu()
            A_dict[(chars[0], chars[1])] = A
    
        print("epoch %d finished in %.2f seconds" % (epoch, (time.time() - epoch_start_t)))
        #print(epoch_total_loss)
        
        avg_loss = epoch_total_loss/k
        print(f"average loss per epoch is {avg_loss}",  flush=True)
        
        avg_losses.append(avg_loss)
        
        f_tpl = open(training_progress_log, 'a')
        f_tpl.write('epoch ' + str(epoch) + ': ' + str(time.time() - epoch_start_t) + '\n')
        f_tpl.close()
    
    avg_losses_cpu = [loss.to('cpu') for loss in avg_losses]
    
    plt.figure()
    plt.plot(range(epoch+1), avg_losses_cpu)
    plt.ylabel('Average loss per epoch')
    plt.xlabel('Epoch')
    plt.title(f"======{cmap[chars[0]]}-{cmap[chars[1]]}======")
    plt.savefig(plot_save_path + f'LARN-origin-TempTrend-{cmap[chars[0]]}-{cmap[chars[1]]}.png')
    
    torch.save(model, model_save_path + model_object_file_name)
    pickle.dump(A_dict, open(A_dict_file_name, 'wb'))
