#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:12:11 2022

@author: yuwenchen
"""

from constants import *
from utils import *
from modules import *

import random
import time
import matplotlib.pyplot as plt

from collections import defaultdict
import pandas as pd
import torch.nn as nn
import pickle 
import torch


random.seed(2022)

CUDA_LAUNCH_BLOCKING=1
print(torch.__version__)


class ARM(nn.Module):

    
    def __init__(self, num_heads, d_word, d_topic, d_noun_hidden, d_char, d_book, d_mix, num_descs, num_chars, tensor_We):
        super().__init__()
        
        
        # dim of descriptors = 30 
        self.num_descs = num_descs
        
        
        self.l_ent = nn.Embedding(num_chars, d_char)
        
        self.l_st = TrainedWordEmbeddingLayer(tensor_We, d_word)

        
        self.l_verb_mfa = TopicVerbMultiHead(num_heads, tensor_We, d_topic, d_word)
        
        #### 
        self.l_noun_mfa = TopicNounMultiHead(num_heads, tensor_We, d_topic, d_word, d_noun_hidden)
        

        # V_final
        self.l_mix = MixingLayer_Concat(d_topic, d_noun_hidden, d_char, d_mix)


        # da (# of examples, 30)
        self.l_rels = DistributionLayer(d_mix, num_descs)
        
        # r_a (# of example, 300)
        self.l_recon = nn.Linear(num_descs, d_word, device=device)

        # init the weights
        self.l_recon.apply(init_weights)
        
    def init_attention(self, A):
        
        # initialize the weights using A (Tensor)
        self.l_verb_mfa.query.weight = nn.Parameter(A)

    ## spans, verb_masked_spans, noun_spans, topic_embs, char1, char2, months
    def forward(self, spans, verb_spans, noun_spans, topic_emb, e1, e2, months):



        e1_tensor = torch.as_tensor(e1, dtype=torch.long, device=device)
        e2_tensor = torch.as_tensor(e2, dtype=torch.long, device=device)
        
        topic_emb = topic_emb.unsqueeze(0).to(dtype=torch.float, device=device)
        
        # NEW
        # V_p
        # verb embedding with attention for topic 
       
        outputs_l_verb_mfa = self.l_verb_mfa(topic_emb, verb_spans)
        
        # NEW
        # V_n
        # noun embedding with attention for topic 
        outputs_l_noun_mfa = self.l_noun_mfa(topic_emb, noun_spans, months, month_info_encode=1)
        #print(f"shape of outputs_l_noun_mfa {outputs_l_noun_mfa}")
      
        # V_ei, V_ej
        outputs_l_e1 = self.l_ent(e1_tensor).expand(len(spans), -1)
        outputs_l_e2 = self.l_ent(e2_tensor).expand(len(spans), -1)
        
        
        # NEW
        # V_final -> final shape ([256,300])
        # concat three embeddings, pass into linear layer, then relu
        outputs_l_mix = self.l_mix(outputs_l_verb_mfa, outputs_l_noun_mfa, outputs_l_e1, outputs_l_e2)
        
        # da -> final shape ([256,30])
        # convert label ([256,300]) into probability distribution ([256,30]) over relation description.
        # passed label to linear layer, then softmax.
        outputs_l_rels = self.l_rels(outputs_l_mix)
        
        # r_a -> da is passed into a linear layer
        # reconstruct the information in the text from predicate, entities, nouns
        # final shape ([256,300])
        outputs = self.l_recon(outputs_l_rels)
        
        
        return outputs, outputs_l_rels



# ====== Training ======

if __name__ == "__main__":

    
    # 0. data: 
    
    ############ TEST FOR ALL DATA
    
    import os
    p = os.path.abspath('..')
        
    bmap, cmap, wmap, revmap, span_data, span_size, target_pred_ix_set, We = torch.load(data_path)
    num_chars, num_books, num_traj = len(cmap), len(bmap), len(span_data)
    topic_embs = pd.read_pickle(topic_emb_path)
    
    # NOTE: This is only for US-RU pair. 
    books, chars, all_spans, all_masks, all_months, all_texts = span_data[5]
    
    ############
    
        
    char1 = [chars[0]]
    char2 = [chars[1]]
    pair_name = f"{cmap[chars[0]]}-{cmap[chars[1]]}"
    print(f"We are working on {pair_name}")  
    
    # put all data on GPU
    tensor_We = torch.as_tensor(We, device=device, dtype=torch.float)
    
    
    
    # 3. Use pretrained word embedding to convert label into embeddings.
    # output shape: ([len(input), 300])
    label_generation_layer = TrainedWordEmbeddingLayer(tensor_We, d_word).to(device)

    A_dict = dict()
    
    # 6. train the model 
    
    #for t in range(topic_embs.shape[0]):
    for t in range(4):
        
        print(f"we are now at {t}th topic",  flush=True)
        single_topic_emb = topic_embs[t,:].detach()
        avg_losses =[] 
       
        # 1. model:
        model = ARM(num_heads, d_word, d_topic, d_noun_hidden, d_char, d_book, d_mix, num_descs, num_chars, tensor_We).to(device)
        
        
        # if we have multiple GPUs
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!",  flush=True)
          model = nn.DataParallel(model)
          
        
        # 2. loss and optimizer
        loss_function = Contrastive_Max_Margin_Loss().to(device)
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)
            
      
        for epoch in range(n_epochs):
            print(f"we are now at {epoch}th epoch",  flush=True)
            epoch_start_t = time.time()
            epoch_total_loss = torch.tensor(0., device=device)
      
            zip_list_to_shuffle = list(zip(all_spans, all_masks, all_months))
            random.shuffle(zip_list_to_shuffle)
            shuffled_big_spans, shuffled_big_masks, shuffled_big_months = zip(*zip_list_to_shuffle)
            
            split_indices = [i for i in range(0, len(shuffled_big_spans), batch_size)] # split to batches to fit in memory
            spans_split = np.split(shuffled_big_spans, split_indices)
            masks_split = np.split(shuffled_big_masks, split_indices)
            months_split = np.split(shuffled_big_months, split_indices)
            
            
            # ========== for each batch ========== 
            
            for k, (spans, masks, months) in enumerate(zip(spans_split, masks_split, months_split)):
      
                #print(f"we are now at {k}th iteration", flush=True)
                #iteration_start_t = time.time()
      
                # first span is empty so we need to continue after it 
                if len(spans) < 1:
                    continue
      
                model.zero_grad()
      
                if (chars[0], chars[1]) in A_dict:
                    model.init_attention(A_dict[(chars[0], chars[1])].to(device))
                
                else:
                    nn.init.xavier_uniform_(model.l_noun_mfa.linear.weight)
      
                # only for pred span
                drop_masks = (np.random.rand(*(masks.shape)) < (1 - p_drop)).astype('float32')
                
      
                verb_masked_spans = []
                noun_spans = []
                pos_masked_spans = []
      
                # ========== for each article  ==========
               
                for span, mask, drop_mask in zip(spans, masks, drop_masks):
      
                  verb_span = []
                  pos_masked_span = []
                  noun_span = []
      
                  for i in range(len(span)):
      
                    if mask[i] == predicate_ix:
                      pos_masked_span.append(span[i])
                    
                      if drop_mask[i] == 1:
                        verb_span.append(span[i])
                    
                    if mask[i] == noun_ix:
                      noun_span.append(span[i])
      
                  pos_masked_spans.append(pos_masked_span)
                  verb_masked_spans.append(verb_span)
                  noun_spans.append(noun_span)
      
                
                
                # ============ generate negative example ===============
                neg_spans, neg_masks = generate_negative_samples(num_traj, span_size,
                                                                  num_negs, span_data)
                neg_masked_spans = []
        
                # go through each 15 negative example 
                for span, mask in zip(neg_spans, neg_masks):
                  neg_masked_span = []
                  for i in range(len(span)):
                    if mask[i] == predicate_ix: 
                      neg_masked_span.append(span[i])
                    # neg_masked_span = [span[i] for i in range(len(span)) if mask[i] == predicate_ix or mask[i] == verb_ix]
      
                  neg_masked_spans.append(neg_masked_span)
                    
        
                # ================ Training ========================
                outputs, outputs_l_rels  = model(spans, verb_masked_spans, noun_spans, single_topic_emb, char1, char2, months)
                pos_labels = label_generation_layer(pos_masked_spans)
                neg_labels = label_generation_layer(neg_masked_spans)
      
                R = torch.t(model.l_recon.weight)
      
                loss = loss_function(outputs, pos_labels, neg_labels, len(spans), R, eps, d_word)
                
                loss.backward()
                optimizer.step()         
                
                epoch_total_loss += loss.item()
                #print("iteration %d finished in %.2f seconds" % (k, (time.time() - iteration_start_t)), flush=True)
                
                #A = model.l_tva.query.weight.detach().cpu()
                #A_dict[t] = A
                
            print("epoch %d finished in %.2f seconds" % (epoch, (time.time() - epoch_start_t)), flush=True)
            
            avg_loss = epoch_total_loss/k
            print(f"average loss per epoch is {avg_loss}",  flush=True)
            
            avg_losses.append(avg_loss)
            
            # 
            training_progress_log = f'./outputs/training-progress-log-ARM-{cmap[chars[0]]}-{cmap[chars[1]]}-topic{t}.txt'
            f_tpl = open(training_progress_log, 'a')
            f_tpl.write('epoch ' + str(epoch) + ': ' + str(time.time() - epoch_start_t) + '\n')
            f_tpl.close()
            
        print(f"we save the model of {t}th topic", flush=True)
        torch.save(model, model_save_path + f"trained-model-ARM-{cmap[chars[0]]}-{cmap[chars[1]]}-topic{t}.pt")
        
        avg_losses_cpu = [loss.to('cpu') for loss in avg_losses]
        
        plt.figure()
        plt.plot(range(epoch+1), avg_losses_cpu)
        plt.ylabel('Average loss per epoch')
        plt.xlabel('Epoch')
        plt.title(f"Topic{t}")
        plt.show()
        
        print("We save the model loss.")
        plt.savefig(plot_save_path + f'Loss-{cmap[chars[0]]}-{cmap[chars[1]]}-{t}.png')
        
        # save each topic individually 
        
        #pickle.dump(A_dict, open(f'weights-topic{t}.pt', 'wb'))
        
        
              
              