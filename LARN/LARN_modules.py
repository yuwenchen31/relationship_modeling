from LARN_constants import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2018)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainedWordEmbeddingLayer(nn.Module):

    
    def __init__(self, trained_we_tensor, d_word):
        super(TrainedWordEmbeddingLayer, self).__init__()
        
        # create embedding instance using 'trained_we_tensor' Floattensor which is the weights for the embedding
        self.we = nn.Embedding.from_pretrained(trained_we_tensor)
        
        # dim of embedding 
        self.d_word = d_word

     
    def forward(self, spans):
        
        # spans = (trained_mask_spans = only predicate ids), ()
        # create zero tensor of shape (num_word x emb dim) and put it to device 
        # outputs shape = (len(train_masked_spans) x 300)
        outputs = torch.zeros([len(spans), self.d_word], dtype=torch.float).to(device)
        #print("output shape of word embedding layer:", outputs.shape)


        # go through each examples in spans [[23435], [2345, 5667],...]
        # i = 0,1,2,..255 or depends on the batch size
        for i, span in enumerate(spans):
            
            # span = [23435]
            # if there exists a predicate in this example
            if len(span) > 0:
                
                # input = (torch.LongTensor(span).to(device)) = e.g., tensor([ 88139, 118390])
                # input it to embedding layer -> self.we(input) -> shape (2,300) -> 2 predicates and each has 300 dimensions
                # torch.sum (input, dim=0) = sum over rows = shape is (# of columns)
                # output[0] = sum predicate embedding for sequence 1
                # output[1] = sum predicate embedding for sequence 2 ... 
                outputs[i] = torch.sum(self.we(torch.LongTensor(span).to(device)), 0)
                
        # outputs shape = (len(train_masked_spans), 300)
        return outputs

# encoder, decoder attention
# "What is the relevance between query and key?"
# Q: (1,336)
# K: tanh(linear(V)) - (# of nouns, 336) ()

# The composed answer: attention is the weighted sum of values
# V: noun embeddings - (# of nouns, 336)
class NounAttentionLayer_SingleQuery(nn.Module):

    def __init__(self, trained_we_tensor, d_word, d_noun_hidden, d_desc):
        super(NounAttentionLayer_SingleQuery, self).__init__()
        
       
        # trained_we_tensor is the pretrained weight for the embedding. Can also use word2vec as the weights 
        # self.we is the embedding layer: self.we(input)
        self.we = nn.Embedding.from_pretrained(trained_we_tensor).to(device)
        self.d_word = d_word
        
        self.d_noun_hidden = d_noun_hidden
        
        # number of descriptors 
        self.d_desc = 1
        
        # Initialize a linear layer that will have the specified input/output shape
        # nn.Linear(size of each input example, size of each output example), bias=False -> do not set bias
        # use each entity pair as key to compute attention weights
        self.linear_key = nn.Linear(d_noun_hidden, d_noun_hidden, bias=False)
        
        # Initialize the learnable weights of a single layer 
        # Fills the input Tensor with values using uniform distribution (aka Glorot initialization)
        # NOTE: why this initilization? "xavier_uniform_"
        nn.init.xavier_uniform_(self.linear_key.weight)
        
        # dictionary (numbers) of embedding = d_desc = 1. Its dimension = d_noun_hidden = 336
        self.query = nn.Embedding(d_desc, d_noun_hidden)

    def forward(self, noun_spans, months, month_info_encode=1):
        
        # noun_spans = [[1123,345], [2345], ...] -> 256 examples in one batch, each is a list with # of nouns 
        # d_noun_hidden*d_desc(1) = 336*1 = 336 dim
        # outputs.shape = (256 examples, 336 dim) -> each example has 336 dim
        outputs = torch.zeros([len(noun_spans), self.d_noun_hidden * self.d_desc], dtype=torch.float).to(device)

        # e.g., i =0 ,
        # noun_span = [23445,23445]
        # month = 4
        # i = 0,1,2,...,255
        for i, (noun_span, month) in enumerate(zip(noun_spans, months)):
            
            # if there are nouns in the sentence
            if len(noun_span) > 0:
                
                # convert noun into word embedding -> each noun will have 300 dim embeddings
                # shape = (# of nouns, 300 dim)
                noun_part_mat = self.we(torch.LongTensor(noun_span).to(device))
                
                # len_part_mat = len of noun_span = how many nouns are there
                # d_noun_hidden (336) - d_word (300) = 36 dim (12 months * 3 years) for time encoding 
                # shape = (# of nouns, 36 months)
                month_part_mat = torch.zeros([len(noun_part_mat), self.d_noun_hidden - self.d_word],
                                             dtype=torch.float).to(device)
                
                
                # j = 0,1,2,.. to (# of nouns)-1
                # month = any number between 1,2,3.., 30
                # j = 0 -> the first noun -> first row 
                # month = 15 -> the 15th month -> the 15th column 
                # fill in 1 when it is published month
                for j in range(len(month_part_mat)):
                    month_part_mat[j][month] = month_info_encode

                # V: concat both along the dim=1. Shape (# of nouns, 336)
                # concat the info of noun & time
                hidden_mat = torch.cat((noun_part_mat, month_part_mat), 1)
                
                # you can get Q,K,V by putting the input sentence through linear layer 
                # K: (# of nouns, 336 dim)
                key_mat = self.linear_key(hidden_mat)
                key_mat = torch.tanh(key_mat)
                
                # d_desc = 1 = num of descriptors

                # Q: shape (1, 336 dim)
                query_mat = self.query(torch.LongTensor([j for j in range(self.d_desc)]).to(device))

                # compute attention weights = alpha_mat
                # how relevant is the key (noun+published time) to the query (descriptor)?  
                # torch.t (key_mat) -> transpose key_mat
                # matrix multiplication of "query_mat" and transposed "key_mat" -> the relevance between query and key 
                # shape (1 queries, # of nouns)
                alpha_mat = torch.mm(query_mat, torch.t(key_mat))
                
                # NOTE: maybe missing the scaled-dot-product?????
                alpha_mat = F.softmax(alpha_mat, dim=1)

                # context = dot product (torch.mm) of attention weights (alpha) and values (hidden_mat: shape (2,336))
                # shape (1, 336)
                result_mat = torch.mm(alpha_mat, hidden_mat)
                
                # flatten the result_mat to 1D shape -> 1*336=336
                result_cat = result_mat.view(self.d_noun_hidden * self.d_desc)
                
                # put the flatten tensor into outputs shape (256, 336)
                outputs[i] = result_cat

        return outputs

class MixingLayer(nn.Module):

    def __init__(self, d_word, d_ent, d_meta, d_mix):
        super(MixingLayer, self).__init__()
        
        self.linear_word = nn.Linear(d_word, d_mix)
        self.linear_ent = nn.Linear(d_ent, d_mix, bias=False)
        self.linear_meta = nn.Linear(d_meta, d_mix, bias=False)
        self.func = F.relu
        nn.init.xavier_uniform_(self.linear_word.weight)
        nn.init.constant_(self.linear_word.bias, 0.)
        nn.init.xavier_uniform_(self.linear_ent.weight)
        nn.init.xavier_uniform_(self.linear_meta.weight)

    def forward(self, input_word, input_ent1, input_ent2, input_meta):
        return self.func(self.linear_word(input_word) + self.linear_ent(input_ent1 + input_ent2)
                         + self.linear_meta(input_meta))
    

class MixingLayer_Attention_SingleQuery_Concat(nn.Module):


    # (300, 336, 50, 300)
    def __init__(self, d_word, d_noun_hidden, d_ent, d_mix):
        
        super(MixingLayer_Attention_SingleQuery_Concat, self).__init__()

        # input node #: 300 + 336 + 50 = 686. Output nodes: 300
        self.linear_cat = nn.Linear(d_word + d_noun_hidden + d_ent, d_mix)

        self.func = F.relu

        # initialize the weights for this single layer
        nn.init.xavier_uniform_(self.linear_cat.weight)

        # init.constant_ ('param', 'const') -> use the 'const' value to init 'param'
        # initialize the bias parameter of linear layer as 0
        nn.init.constant_(self.linear_cat.bias, 0.)
        self.d_noun_hidden = d_noun_hidden

    # (outputs_l_st, outputs_l_noun, outputs_l_e1, outputs_l_e2)
    # (256,300) (256,336) (256,50) (256,50)
    def forward(self, input_word, input_noun_cat, input_ent1, input_ent2):

        # .view -> reshape the elements of the tensor -> if (a,b), then a*b has to be the same # of tensor elements
        # len(outputs_l_noun)= 256
        # (256,336) -> [(256,1,336)]
        noun_mats = input_noun_cat.view([len(input_noun_cat), 1, self.d_noun_hidden])
        
        from_noun_mats = []

        # index_select(tensor, dim, index"tensor contains the index")
        # e.g., noun_mats = torch.tensor([[[ 1,  2,  3]],
        #                     [[ 1,  2,  3]],
        #                     [[ 7,  8,  9]],
        #                     [[13, 14, 15]]])
        # noun_mats.shape -> torch.Size([4, 1, 3])
        # dim = 1 -> select along the row. torch.LongTensor([0]) -> select the 1st 'row'
        # Because there is only 1 row for each example, it returns the same tensor. (len(input_noun_cat)=256, d_noun_hidden=336)
        # reshape it into (256, 336). append into a list
        from_noun_mats.append(torch.index_select(noun_mats, 1, torch.LongTensor([0]).to(device)).view([len(input_noun_cat), self.d_noun_hidden]))

        # e.g., from_noun_mats = [tensor([[ 1,  2,  3,...],
        #                                 [ 1,  2,  3,...],
        #                                 [ 7,  8,  9,...],
        #                                 [13, 14, 15,...]])]
        # concat along dim = 1, along the row.
        # torch.Size([4, 3])
        # e.g., from_nouns = tensor([[ 1,  2,  3],
        #                           [ 1,  2,  3],
        #                           [ 7,  8,  9],
        #                           [13, 14, 15]])
        from_nouns = torch.cat(from_noun_mats, dim=1)

        # input_ent1 + input_ent2 = add element-wise -> shape ([256,50])
        # cat along each row -> for each article, concat those embeddings
        # cat_inputs.shape -> torch.Size([256, 686])
        # 300 + 336 + 50 = 686
        cat_inputs = torch.cat([input_word, from_nouns, input_ent1 + input_ent2], dim=1)
                
        # V_final
        # put into linear layer and then activation function relu. -> ([256, 300]).
        # relu = max(0,x)
        # if it is negative then it is 0, positive is 1. -> LABEL NOTATION!!
        return self.func(self.linear_cat(cat_inputs))

class LinearRNN(nn.Module):

    def __init__(self, d_input, d_hidden):
        super(LinearRNN, self).__init__()
        self.d_hidden = d_hidden
        self.begin = True
        self.hidden = torch.zeros([self.d_hidden], dtype=torch.float).to(device)
        self.linear_input = nn.Linear(d_input, d_hidden, bias=False)
        self.linear_hidden = nn.Linear(d_hidden, d_hidden, bias=False)
        self.func = F.softmax
        self.alpha = 0.5 # inherited from RMN

    def forward(self, inp, hid):
        from_inp = self.linear_input(inp)
        if self.begin:
            output = self.func(from_inp, dim=0)
            self.begin = False
        else:
            from_hid = self.linear_hidden(hid)
            output = self.func(from_inp + from_hid, dim=0)
            output = output * self.alpha + hid * (1 - self.alpha)
        return output, output
    
class DistributionLayer(nn.Module):

    # d_input = d_mix = 300. d_hidden = d_desc = 30.
    def __init__(self, d_input, d_hidden):
        super(DistributionLayer, self).__init__()
        
        # nn.Linear(in_features=300, out_features=30). No bias.
        self.linear_input = nn.Linear(d_input, d_hidden, bias=False)

        # init the weights of linear layer
        nn.init.xavier_uniform_(self.linear_input.weight)

        # activation function as softmax. (making the value for each article between 0-1)
        self.func = F.softmax

    def forward(self, inps):

        # final shape: ([256,30])
        from_inps = self.linear_input(inps)
        
        # Along each row (article), apply softmax.
        # We can get the probability distribution over 30 relation descriptors for each article.
        # final shape: ([256,30])
        outputs = self.func(from_inps, dim=1)
        
        # da -> ([256,30])
        return outputs

class Contrastive_Max_Margin_Loss(nn.Module):

    def __init__(self):
        super(Contrastive_Max_Margin_Loss, self).__init__()
    
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
                norm_outputs[i] = torch.zeros([d_word], dtype=torch.float).to(device)

        # Unit vector of predicates embedding in each article
        norm_pos_labels = pos_labels / torch.norm(pos_labels, 2, 1, True)

        # check if any element is NaN
        nan_masks = torch.isnan(norm_pos_labels)

        # i = 0,1,2,..255
        for i in range(len(nan_masks)):

            # if in one article, there is at least 1 NaN (True). We replace article embedding as tensor 0
            if torch.sum(nan_masks[i]) > 0:

                # norm_pos_labels.shape = ([256,300])
                norm_pos_labels[i] = torch.zeros([d_word], dtype=torch.float).to(device)

        norm_neg_labels = neg_labels / torch.norm(neg_labels, 2, 1, True)
        nan_masks = torch.isnan(norm_neg_labels)
        
        for i in range(len(nan_masks)):
            if torch.sum(nan_masks[i]) > 0:

                # norm_neg_labels.shape = ([15, 300])
                norm_neg_labels[i] = torch.zeros([d_word], dtype=torch.float).to(device)


        # torch.sum (input, dim, keepdim) -> sum over dim=1(along the row)
        #
        correct = torch.sum(norm_outputs * norm_pos_labels, 1, True)
        wrong = torch.mm(norm_outputs, torch.t(norm_neg_labels))

        loss = torch.sum(torch.max(torch.zeros(traj_length).to(device),
                                   torch.sum(1. - correct + wrong, 1)))

        # unit vector of R
        # ([30,300])
        norm_R = R / torch.norm(R, 2, 1, True)

        # X = RRT - I
        # torch.eye: with 1s on diagonal, and 0s elsewhere. -> torch.eye(norm_R.shape[0])= torch.eye(30)= (30,30) square with 1s on the diagonal
        ortho_p = eps * torch.sum((torch.mm(norm_R, torch.t(norm_R)) - torch.eye(norm_R.shape[0]).to(device)) ** 2)

        loss += ortho_p.to(device)

        return loss
