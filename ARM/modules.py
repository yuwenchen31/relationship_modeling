from constants import *
from utils import * 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import math
from flair.embeddings import TransformerDocumentEmbeddings,SentenceTransformerDocumentEmbeddings, WordEmbeddings, DocumentRNNEmbeddings
from flair.data import Sentence
import pickle
torch.manual_seed(2022)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class TopicEmbeddingLayer(nn.Module):

    # TODO: use sentence transfomer
    def __init__(self, emb_model='all-mpnet-base-v2'): #emb_model='roberta-base'
        super().__init__()
        #self.document_embeddings = SentenceTransformerDocumentEmbeddings(emb_model)
        
        #self.glove_embedding = WordEmbeddings('glove')
        #self.document_embeddings = DocumentRNNEmbeddings([self.glove_embedding], hidden_size=300)
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
        
    # prob: ((# of examples, num_topics))
    # TODO: check how to get d_doc dynamically? 
    def forward(self, texts, probs, d_doc=768):
        
        doc_emb = torch.zeros([len(texts), d_doc], dtype=torch.float64, device=device)
        
        #for i,doc in enumerate(texts):
          #doc = Sentence(doc)
          #self.document_embeddings.embed(doc)
          #doc_emb[i] = doc.embedding
        
        for i,doc in enumerate(texts):
          embeddings = self.model.encode(doc)
          doc_emb[i] = torch.from_numpy(embeddings)

        probs = torch.from_numpy(probs).to(dtype=torch.float64, device=device)
       
        # # weighted topic embeddings: (n_topic, 128) 
        topic_emb = torch.matmul(torch.t(probs),doc_emb)

        return topic_emb

    
# New 
class TopicVerbAttention(nn.Module):

    def __init__(self, trained_we_tensor, d_topic, d_word):
        
        super().__init__()
        
        # TODO: change this to bert embedding 
        self.v = nn.Embedding.from_pretrained(trained_we_tensor)
        self.d_topic = d_topic
        self.d_word = d_word
        # Initialize a linear layer that will have the specified input/output shape
        # nn.Linear needs input as tensor.float
        self.k = nn.Linear(trained_we_tensor.shape[1], d_topic, device=device)
        self.q = nn.Linear(d_topic, d_topic, device=device)
        nn.init.xavier_uniform_(self.k.weight)

        #self.query = nn.Embedding(d_topic, d_topic)
        
    def forward(self, topic_emb, verb_span):
        
        # (# of batches, 768)
        outputs = torch.zeros([len(verb_span), self.d_word], dtype=torch.float64, device=device)
        

        # go through each docs
        for i, verb in enumerate(verb_span): 
            
            if len(verb_span) > 0:
        
                # V (# of verb, 300)
               
                value = self.v(torch.as_tensor(verb, dtype=torch.long,device=device))
                key = self.k(value)
                query = self.q(topic_emb)
                result = scaled_dot_product_attention(query, key, value)
                outputs[i] = result

        # shape (# of batch, dim of topics=768)
        return outputs    


# New
class TopicNounAttention(nn.Module):

   
    def __init__(self, trained_we_tensor, d_topic, d_word, d_noun_hidden):
        
        super().__init__()
        
        
        # TODO: change this to bert embedding 
        self.v = nn.Embedding.from_pretrained(trained_we_tensor).to(device)
        self.d_topic = d_topic
        self.d_word = d_word
        self.d_noun_hidden = d_noun_hidden
        self.k = nn.Linear(d_noun_hidden, d_topic, device=device)
        self.q = nn.Linear(d_topic, d_topic, device=device)
        
        #self.query = nn.Embedding(d_topic, d_topic)
     
        nn.init.xavier_uniform_(self.k.weight)
        
 
    def forward(self, topic_emb, noun_spans, months, month_info_encode=1):
        
       # (# of batches, 336)
        outputs = torch.zeros([len(noun_spans), self.d_noun_hidden], dtype=torch.double, device=device)
        
        
        for i, (noun_span, month) in enumerate(zip(noun_spans, months)):
            
            # if there are nouns in the sentence
            if len(noun_span) > 0:

                noun_part_mat = self.v(torch.as_tensor(noun_span, dtype=torch.long, device=device))
                month_part_mat = torch.zeros([len(noun_part_mat), self.d_noun_hidden - self.d_word],
                                             dtype=torch.float, device=device)
                            
                for j in range(len(month_part_mat)):
                    month_part_mat[j][month] = month_info_encode

                # V: concat info of noun & time along the dim=1. 
                # Shape (# of nouns, 336)
                value = torch.cat((noun_part_mat, month_part_mat),1).to(device)

                # K: (# of nouns, 336 dim)
                key = self.k(value)
                query = self.q(topic_emb)
                
                result_mat = scaled_dot_product_attention(query, key, value)
                
              
                outputs[i] = result_mat

        # (# of batch, d_noun_hidden=336)
        return outputs

# New: adapted from (https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51)
class TopicVerbMultiHead(nn.Module):
        
        def __init__ (self, num_heads, trained_we_tensor, d_topic, d_word):
            
            super().__init__()
            self.heads = nn.ModuleList(
                [TopicVerbAttention(trained_we_tensor, d_topic, d_word) for _ in range(num_heads)]
                )
            self.linear = nn.Linear(num_heads * trained_we_tensor.shape[1], d_topic, device=device)
            
        def forward(self, topic_emb, verb_span): 
        
            outputs = [h(topic_emb, verb_span) for h in self.heads]    
            return self.linear(torch.cat(outputs, dim=-1).to(torch.float))
        
# New
class TopicNounMultiHead(nn.Module):
    
        def __init__ (self, num_heads, trained_we_tensor, d_topic, d_word, d_noun_hidden):
            
            super().__init__()
            self.heads = nn.ModuleList(
                [TopicNounAttention(trained_we_tensor, d_topic, d_word, d_noun_hidden) for _ in range(num_heads)]
                )
            self.linear = nn.Linear(num_heads * d_noun_hidden, d_noun_hidden, device=device)

        def forward(self, topic_emb, noun_spans, months, month_info_encode=1): 
        
            head_output = [h(topic_emb, noun_spans, months, month_info_encode=1) for h in self.heads]    
            final_outputs = self.linear(torch.cat(head_output, dim=-1).to(torch.float))

            return final_outputs

# Modified from LARN
class MixingLayer_Concat(nn.Module):

    def __init__(self, d_topic, d_noun_hidden, d_ent, d_mix):

        super().__init__()
        self.d_noun_hidden = d_noun_hidden
        self.d_topic = d_topic
        
        # input node #: 300 + 336 + 50 = 686. Output nodes: 300
        self.linear_cat = nn.Linear(d_topic + d_noun_hidden + d_ent, d_mix, device=device)
        self.func = F.relu

        # initialize the weights for this single layer
        nn.init.xavier_uniform_(self.linear_cat.weight)

        # init.constant_ ('param', 'const') -> use the 'const' value to init 'param'
        # initialize the bias parameter of linear layer as 0
        nn.init.constant_(self.linear_cat.bias, 0.)

    # (outputs_l_tva, outputs_l_noun, outputs_l_e1, outputs_l_e2)
    # (256,300) (256,336) (256,50) (256,50)
    def forward(self, input_verb , input_noun, input_ent1, input_ent2):

        noun_mats = input_noun.view([len(input_noun), 1, self.d_noun_hidden])
        from_noun_mats = []
        from_noun_mats.append(torch.index_select(noun_mats, 1, torch.as_tensor([0], dtype=torch.long, device=device)).view([len(input_noun), self.d_noun_hidden]))
        from_nouns = torch.cat(from_noun_mats, dim=1)
        cat_inputs = torch.cat([input_verb, from_nouns, input_ent1 + input_ent2], dim=1)
        outputs = self.func(self.linear_cat(cat_inputs.to(torch.float)))

        # (# of example, 300)
        return outputs


class TrainedWordEmbeddingLayer(nn.Module):

    def __init__(self, trained_we_tensor, d_word):
        super().__init__()
        
        # create embedding instance using 'trained_we_tensor' Floattensor which is the weights for the embedding
        self.we = nn.Embedding.from_pretrained(trained_we_tensor).to(device)
        
        # dim of embedding 
        self.d_word = d_word

    def forward(self, spans):
        
        # spans = (trained_mask_spans = only predicate ids), ()
        # create zero tensor of shape (num_word x emb dim) and put it to device 
        # outputs shape = (len(train_masked_spans) x 300)
        outputs = torch.zeros([len(spans), self.d_word], dtype=torch.float, device=device)

        # go through each examples in spans [[23435], [2345, 5667],...]
        # i = 0,1,2,..255 or depends on the batch size
        for i, span in enumerate(spans):
            
            # span = [23435]
            # if there exists a predicate/noun in this example
            if len(span) > 0:
                outputs[i] = torch.sum(self.we(torch.as_tensor(span,dtype=torch.long,device=device)), 0)
                
        # outputs shape = (len(train_masked_spans), 300)
        return outputs
    



# change this to BERT? 
class DistributionLayer(nn.Module):
    
    # d_input = d_mix = 300. d_hidden = d_desc = 30.
    def __init__(self, d_input, d_output):
        super().__init__()
        
      
        # nn.Linear(in_features=300, out_features=30). No bias.
        self.linear_input = nn.Linear(d_input, d_output, bias=False, device=device)
            
        # activation function as softmax. (making the value for each article between 0-1)
        self.func = F.softmax
        # init the weights of linear layer
        nn.init.xavier_uniform_(self.linear_input.weight)
        
    def forward(self, inps):

        # final shape: ([256,30])
        from_inps = self.linear_input(inps)
        
        # Along each row (article), apply softmax.
        # We can get the probability distribution over 30 relation descriptors for each article.
        # final shape: ([256,30])
        outputs = self.func(from_inps, dim=1)

        # da -> ([# of example,30])
        return outputs


class Contrastive_Max_Margin_Loss(nn.Module):

    def __init__(self):
        super().__init__()
    
    # (outputs, pos_labels, neg_labels, len(spans), R, eps, d_word)
    # outputs = after l_con = (256,300)
    # pos_labels = predicate embedding per article = (256,300)
    # neg_labels = predicate embedding per negative(random) articles = (15,300)
    # len(spans) = 256
    # R = transpose weight of linear(da) = (30,300)
    # eps = lambda (in paper) = descriptor matrix orthogonal penalty weight = 1e-1

    def forward(self, outputs, pos_labels, neg_labels, traj_length, R, eps, d_word):

        # NOTE: change torch.norm to torch.vector_norm (I need to update pytorch version)
        # torch.norm -> calculate vector length
        # final shape = ([256,300])
        # unit vector of outputs = a vector length=1  = vector / vector length
        norm_outputs = outputs / torch.norm(outputs, 2, 1, True)

        # torch.isnan -> when element is NaN: True, else: False
        # nan_masks = boolean tensor ([256,300])
        # tensor([[False, False, False,  ..., False, False, False],
        #         [False, False, False,  ..., False, False, False],
        #         [False, False, False,  ..., False, False, False],...])
        nan_masks = torch.isnan(norm_outputs)

        
        # i = 0,1,2,.. 255
        for i in range(len(nan_masks)):

            # sum of nan_masks of each article: if all of them are False, it is tensor([0])
            # so if it is > 0: there is at least 1 NaN. Then we replace the whole article embedding as 300 dim of 0
            if torch.sum(nan_masks[i]) > 0:
                norm_outputs[i] = torch.zeros([d_word], dtype=torch.float, device=device)

        # Unit vector of predicates embedding in each article
        norm_pos_labels = pos_labels / torch.norm(pos_labels, 2, 1, True)

        # check if any element is NaN
        nan_masks = torch.isnan(norm_pos_labels)

        # i = 0,1,2,..255
        for i in range(len(nan_masks)):

            # if in one article, there is at least 1 NaN (True). We replace article embedding as tensor 0
            if torch.sum(nan_masks[i]) > 0:

                # norm_pos_labels.shape = ([256,300])
                norm_pos_labels[i] = torch.zeros([d_word], dtype=torch.float, device=device)

        norm_neg_labels = neg_labels / torch.norm(neg_labels, 2, 1, True)
        nan_masks = torch.isnan(norm_neg_labels)
        
        for i in range(len(nan_masks)):
            if torch.sum(nan_masks[i]) > 0:

                # norm_neg_labels.shape = ([15, 300])
                norm_neg_labels[i] = torch.zeros([d_word], dtype=torch.float, device=device)


        # torch.sum (input, dim, keepdim) -> sum over dim=1(along the row)
        #
        
        correct = torch.sum(norm_outputs * norm_pos_labels, 1, True)
        wrong = torch.mm(norm_outputs, torch.t(norm_neg_labels))

        loss = torch.sum(torch.max(torch.zeros(traj_length, device=device),
                                   torch.sum(1. - correct + wrong, 1)))

        # unit vector of R
        # ([30,300])
        norm_R = R / torch.norm(R, 2, 1, True)

        # X = RRT - I
        # torch.eye: with 1s on diagonal, and 0s elsewhere. -> torch.eye(norm_R.shape[0])= torch.eye(30)= (30,30) square with 1s on the diagonal
        ortho_p = eps * torch.sum((torch.mm(norm_R, torch.t(norm_R)) - torch.eye(norm_R.shape[0], device=device)) ** 2)

        loss += ortho_p

        return loss
