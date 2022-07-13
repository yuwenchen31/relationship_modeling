#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import time
from collections import Counter
from collections import defaultdict
import pandas as pd 
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import numpy as np

random.seed(2022)

from LARN_constants import *
from LARN_utils import *
from LARN_modules import *

model_id = 'LARN'

bmap, cmap, wmap, revmap, span_data, span_size, target_pred_ix_set, We = torch.load("/tudelft.net/staff-bulk/ewi/insy/II/yuwenchen/larn_data_smaller.pt")
num_chars, num_books, num_traj = len(cmap), len(bmap), len(span_data)

# book, chars, topic_span, topic_mask, topic_month
# # of topic_span, topic_mask, topic_month = 10
# TODO 
larn_input = pd.read_pickle("./LARN-us-ru-4tp.pk")


descriptor_log = 'descriptors_model_' + str(model_id) + '.log'
trajectory_log = 'trajectories_model_' + str(model_id) + '.log'
desc_sample_file_name = 'desc_selected_sample_dict_model_' + str(model_id) + '.pkl'
attn_sample_file_name = 'attn_selected_sample_dict_model_' + str(model_id) + '.pkl'
training_progress_log = 'training_progress_log_' + str(model_id) + '.txt'
model_object_file_name = 'trained_model_' + str(model_id) + '.pt'
A_dict_file_name = 'A_dict_' + str(model_id) + '.pkl'


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
        
        # initialize the weights using A (Tensor)
        self.l_noun.query.weight = nn.Parameter(A)
    
    def forward(self, spans, noun_spans, e1, e2, months):
    
        # e.g.,
        # spans=train_masked_spans= [[140897], [23455, 907655],... []] 256 articles
        # noun_spans=[[2334], [4557,98450],..] 256 articles
        # e1 = [5], e2 = [9]
        # month=[23 25 9...] 256 items
    
        e1_tensor = torch.LongTensor(e1).to(device)
        e2_tensor = torch.LongTensor(e2).to(device)
        
        # V_p
        # train_masked_spans is the input -> use drop_mask to drop some words (prob < 0.5) in the article
        # outputs_l_st: shape (len(train_masked_spans) x 300)
        # for each article, there is a 300 dimension predicate embedding (sum of all predicates in the article)
        outputs_l_st = self.l_st(spans)
    
        # V_n
        # noun_spans, months into 'NounAttentionLayer_SingleQuery' 
        # l_noun: shape (256, 10080) -> encodes 30 desc & noun emb & one-hot time info
        outputs_l_noun = self.l_noun(noun_spans, months)
        
        # V_ei, V_ej
        # .expand -> expand (copy) the tensor to the size. "-1" means do not change that dim
        # len(spans) = len(train_masked_spans) = 256
        # l_ent(e1_tensor) -> (1, 50)
        # expand(256,-1) -> shape of outpouts_l_e1 (256, 50) -> makes the entity embedding the same for each article
        # [[-1.234, -0.987,...],[-1.234, -0.987,...],,.. 256th]
        outputs_l_e1 = self.l_ent(e1_tensor).expand(len(spans), -1)
        outputs_l_e2 = self.l_ent(e2_tensor).expand(len(spans), -1)
        
        # V_final -> final shape ([256,300])
        # concat three embeddings, pass into linear layer, then relu
        outputs_l_mix = self.l_mix(outputs_l_st, outputs_l_noun, outputs_l_e1, outputs_l_e2)
        
        # da -> final shape ([256,30])
        # convert label ([256,300]) into probability distribution ([256,30]) over relation description.
        # passed label to linear layer, then softmax.
        outputs_l_rels = self.l_rels(outputs_l_mix)
        
        # da is passed into a linear layer
        # reconstruct the information in the text from predicate, entities, nouns
        # final shape ([256,300])
        outputs = self.l_recon(outputs_l_rels)
        
        
        return outputs, outputs_l_rels


            

if __name__ == "__main__":
    

    # 3. Use pretrained word embedding to convert label into embeddings.
    # output shape: ([len(input), 300])
    label_generation_layer = TrainedWordEmbeddingLayer(torch.FloatTensor(We), d_word).to(device)
    
    book, chars, all_topic_spans, all_topic_masks, all_topic_months = larn_input
    
    A_dict = dict()
    

    # 3. train the model 
    
    char1 = [chars[0]]
    char2 = [chars[1]]
    
    
    # TODO: for first 3 topics
    for tpnr in range(4): 
        
        print(f"We are now at {tpnr} topics.")
        avg_losses =[] 
        
        # 1. model: in LARN (d_word, d_noun_hidden, d_ent, d_meta, d_mix, d_desc, n_ent, n_meta, We)
        model = LARN(d_word, d_noun_hidden, d_char, d_book, d_mix, num_descs, num_chars, num_books, We).to(device)
        
        # 2. loss and optimizer: in Module.py (Contrastive_Max_Margin_Loss())
        loss_function = Contrastive_Max_Margin_Loss().to(device)
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)
            
        big_spans = all_topic_spans[tpnr]
        big_masks = all_topic_masks[tpnr]
        big_months = all_topic_months[tpnr]
        
        for epoch in range(n_epochs):
            print(f"we are now at {epoch} epoch")
            epoch_start_t = time.time()
            
            # epoch_total_loss -> tensor(0.)
            epoch_total_loss = torch.tensor(0.).to(device)
            random.shuffle(span_data)
            
            
            #for book, chars, big_spans, big_masks, big_months, _ in span_data:
           
                #char1 = [chars[0]] # e.g., [5]
                #char2 = [chars[1]] # e.g., [9]
    
            zip_list_to_shuffle = list(zip(big_spans, big_masks, big_months))
      
            # randomly shuffle the list 
            random.shuffle(zip_list_to_shuffle)
      
            # assign the new shuffled items in 'zip_list_to_shuffle' to new attributes 
            shuffled_big_spans, shuffled_big_masks, shuffled_big_months = zip(*zip_list_to_shuffle)
      
            # split to batches to fit in memory. batch_size = 256 
            # [0, 256, 512, 768,...]
            split_indices = [i for i in range(0, len(shuffled_big_spans), batch_size)] 
      
            
            spans_split = np.split(shuffled_big_spans, split_indices)
            masks_split = np.split(shuffled_big_masks, split_indices)
            months_split = np.split(shuffled_big_months, split_indices)
      
            # ========== for each batch ========== 
            for k, (spans, masks, months) in enumerate(zip(spans_split, masks_split, months_split)):
                
                #print(f"we are now at {k}th iteration", flush=True)
                iteration_start_t = time.time()
      
                # first span is empty so we need to continue after it 
                if len(spans) != batch_size:
                    continue
    
                model.zero_grad()
      
                # if entity pairs are in A_dict, initialize the weights using A_dict[(chars[0], chars[1])]
                if (chars[0], chars[1]) in A_dict:
                    model.init_attention(A_dict[(chars[0], chars[1])].to(device))
      
                else:
                    nn.init.xavier_uniform_(model.l_noun.query.weight)
      
                train_masked_spans = []
                noun_spans = []
      
             
                drop_masks = (np.random.rand(*(masks.shape)) < (1 - p_drop)).astype('float32')
      
      
                # ========== for each article  ==========
             
                for span_index, (span, mask, drop_mask) in enumerate(zip(spans, masks, drop_masks)):
      
                   
                    train_masked_span = [span[i] for i in range(len(span))
                                        if mask[i] == predicate_ix and drop_mask[i] == 1]
      
                   
                    train_masked_spans.append(train_masked_span)
                    noun_span = [span[i] for i in range(len(span)) if mask[i] == noun_ix]
                    noun_spans.append(noun_span)
      
      
                # ========== for each article ==========
                pos_masked_spans = []
      
                # go through each article
                for span_index, (span, mask) in enumerate(zip(spans, masks)):
      
                    # get the word id of predicates
                    pos_masked_span = [span[i] for i in range(len(span)) if mask[i] == predicate_ix]
      
                    # pos_masked_spans = [[pre1, pre2], [pre3],... []] 256 articles within
                    pos_masked_spans.append(pos_masked_span)
      
                # ========================================
      
              
                neg_spans, neg_masks = generate_negative_samples(num_traj, span_size,
                                                                num_negs, span_data)
      
                neg_masked_spans = []
      
                # go through each 15 negative example 
                for span_index, (span, mask) in enumerate(zip(neg_spans, neg_masks)):
      
                    # len(span) = 2102. 
                    # go through each word in article, if word's mask == 2, get the word id from span
                    neg_masked_span = [span[i] for i in range(len(span)) if mask[i] == predicate_ix]
      
                 
                    neg_masked_spans.append(neg_masked_span)
      
                # ========================================
      
      
             
                outputs, _outputs_l_rels = model(train_masked_spans, noun_spans, char1, char2, months)
                pos_labels = label_generation_layer(pos_masked_spans)
                neg_labels = label_generation_layer(neg_masked_spans)
      
                # R = Relation embeddings = weights of linear layer (transpose)
                # weight.shape = ([300,30])
                # R = torch.t(weight) = ([30,300])
                R = torch.t(model.l_recon.weight)
      
                # compare reconstruction & label. Compute loss.
                loss = loss_function(outputs, pos_labels, neg_labels, len(spans), R, eps, d_word)
      
                loss.backward()
                optimizer.step()
                epoch_total_loss += loss
                #print("iteration %d finished in %.2f seconds" % (k, (time.time() - iteration_start_t)), flush=True)
      
                A = model.l_noun.query.weight.detach().cpu()
                A_dict[(chars[0], chars[1])] = A
      
            print("epoch %d finished in %.2f seconds" % (epoch, (time.time() - epoch_start_t)))
            
            avg_loss = epoch_total_loss/k
            print(f"average loss per epoch is {avg_loss}",  flush=True)
            
            avg_losses.append(avg_loss)
            
            # 
            training_progress_log = f'./outputs/training-progress-log-LARN-{cmap[chars[0]]}-{cmap[chars[1]]}-topic{tpnr}.txt'
            f_tpl = open(training_progress_log, 'a')
            f_tpl.write('epoch ' + str(epoch) + ': ' + str(time.time() - epoch_start_t) + '\n')
            f_tpl.close()
            
        print(f"we save the model of {tpnr}th topic", flush=True)
        torch.save(model, model_save_path + f"trained-model-LARN-{cmap[chars[0]]}-{cmap[chars[1]]}-topic{tpnr}.pt")
        
        avg_losses_cpu = [loss.to('cpu') for loss in avg_losses]
        
        plt.figure()
        plt.plot(range(epoch+1), avg_losses_cpu)
        plt.ylabel('Average loss per epoch')
        plt.xlabel('Epoch')
        plt.title(f"Topic{tpnr}")
        plt.show()
        
        print("We save the model loss.")
        plt.savefig(plot_save_path + f'Loss-{cmap[chars[0]]}-{cmap[chars[1]]}-{tpnr}.png')






