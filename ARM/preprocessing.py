#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The following codes are adapted from https://github.com/BoulderDS/LARN/blob/master/data_processing.ipynb

import glob
import os
import re
import spacy
import pickle
from collections import defaultdict
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')
from multiprocessing import Pool

import os 

# parent directory 
p = os.path.abspath('..')


load_path = p + '/data/now-text/'
write_path = p + '/data/preprocessed/'
filter_strs = ['19*', '20*', '21*', '22*'] # change this to '*' for all files
year_base = 20


entity_aliases_dict = {'U.S.': {'U.S.', 'US', 'USA', 'American', 'Trump', 'Biden'},
                       'China': {'China', 'Chinese', 'Xi Jinping'},
                       'Syria': {'Syria', 'Syrian', 'Assad'},
                       'France': {'France', 'French', 'Macron', 'Hollande'},
                       'Germany': {'Germany', 'German', 'Merkel', 'Scholz', 'Steinmeier'},
                       'Canada': {'Canada', 'Canadian', 'Trudeau'},
                       'Russia': {'Russia', 'Russian', 'Putin'},
                       'India': {'India', 'Indian', 'Modi'},
                       'U.K.': {'U.K.', 'UK', 'British', 'Britain', 'Johnson', 'Theresa May'},
                       'Japan': {'Japan', 'Japanese', 'Shinzo Abe', 'Yoshihide Suga', 'Fumio Kishida'},
                       'Iran': {'Iran', 'Iranian', 'Hassan Rouhani', 'Ebrahim Raisi'},
                       'Israel': {'Israel', 'Israeli', 'Netanyahu', 'Naftali Bennett'}}
entities_list = list(entity_aliases_dict.keys())
interest_pair_list = []
for i in range(len(entities_list) - 1):
    for j in range(i + 1, len(entities_list)):
        interest_pair_list.append((entities_list[i], entities_list[j]))
        
nlp = spacy.load('en_core_web_sm')
wnl = WordNetLemmatizer()


SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
REL_PRONS = ["that", "who", "which", "whom", "whose", "where", "when", "what", "why"]

def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs

def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs

def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps:
            moreVerbs.extend([tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs

def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False

def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False

def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs

def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(getObjsFromPrepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None

def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None

def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET" and tok.lower_ not in REL_PRONS]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated

def getAllObjs(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS and tok.lower_ not in REL_PRONS]
    objs.extend(getObjsFromPrepositions(rights))

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs


def findSVOs(tokens):
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    if verbNegated or objNegated: # if negative word, get the antonym
                        neg_v = None
                        found = False
                        for syn in wordnet.synsets(v.lower_):
                            if found:
                                break
                            if syn.pos() != 'v':
                                continue
                            for l in syn.lemmas():
                                if l.antonyms():
                                    neg_v = l.antonyms()[0].name()
                                    found = True
                        if neg_v != None:
                            s_lemma = sub.lemma_ if sub.lemma_ != '-PRON-' else sub.lower_ # spacy's pronoun lemma hack
                            o_lemma = obj.lemma_ if obj.lemma_ != '-PRON-' else obj.lower_
                            svos.append((s_lemma, neg_v, o_lemma))
                    else:
                        s_lemma = sub.lemma_ if sub.lemma_ != '-PRON-' else sub.lower_
                        v_lemma = v.lemma_
                        o_lemma = obj.lemma_ if obj.lemma_ != '-PRON-' else obj.lower_
                        svos.append((s_lemma, v_lemma, o_lemma))
    return svos

def findGenerals(tokens):
    nouns = []
    verbs = []
    adjs = []
    advs = []
    all_words = []
    for tok in tokens:
        t_lemma = tok.lemma_ if tok.lemma_ != '-PRON-' else tok.lower_
        all_words.append(t_lemma)
        if tok.pos_ == "NOUN" or tok.pos_ == "PROPN":
            nouns.append(t_lemma)
        elif tok.pos_ == "VERB" and tok.dep_ != "aux":
            verbs.append(t_lemma)
        elif tok.pos_ == "ADJ":
            adjs.append(t_lemma)
        elif tok.pos_ == "ADV":
            advs.append(t_lemma)
    return nouns, verbs, adjs, advs, all_words


def find_related_entities(line, interest_pair_list):
    related_interest_pairs = []
    for interest_pair in interest_pair_list:
        e1_appears = False
        e2_appears = False
        for e1 in entity_aliases_dict[interest_pair[0]]:
            if e1 in line:
                e1_appears = True
        for e2 in entity_aliases_dict[interest_pair[1]]:
            if e2 in line:
                e2_appears = True
        if e1_appears and e2_appears:
            related_interest_pairs.append(interest_pair)
    return related_interest_pairs


def process_news_article(line, interest_pair_list):
    split_list = re.split(r'<.>', line) # split to paragraphs using <h> and <p>
    
    news_svo_info = defaultdict(list)
    news_nn_info = defaultdict(list)
    news_vb_info = defaultdict(list)
    news_jj_info = defaultdict(list)
    news_rb_info = defaultdict(list)
    news_all_info = defaultdict(list)
    news_samples = defaultdict(list)
    
    if len(split_list) <= 1:
        return 0, news_svo_info, news_nn_info, news_vb_info, news_jj_info, news_rb_info, news_all_info, news_samples
    
    try:
        news_index = int(split_list[0][2:])
    except:
        return 0, news_svo_info, news_nn_info, news_vb_info, news_jj_info, news_rb_info, news_all_info, news_samples
    
    for split in split_list[1:]:
        related_interest_pairs = []
        for interest_pair in interest_pair_list:
            e1_appears = False
            e2_appears = False
            for e1 in entity_aliases_dict[interest_pair[0]]:
                if e1 in split:
                    e1_appears = True
            for e2 in entity_aliases_dict[interest_pair[1]]:
                if e2 in split:
                    e2_appears = True
            if e1_appears and e2_appears: # if paragraph contains both entities
                related_interest_pairs.append(interest_pair)
            
        if len(related_interest_pairs) == 0: # paragraph not containing any entity pair
            continue
            
        doc = nlp(split)
        for sent in doc.sents:
            if '@ @' in sent.text: # broken sentence
                continue
                
            is_target_sent = False
            sent_related_interest_pairs = []
            for interest_pair in related_interest_pairs:
                e1_appears = False
                e2_appears = False
                for e1 in entity_aliases_dict[interest_pair[0]]:
                    if e1 in sent.text:
                        e1_appears = True
                for e2 in entity_aliases_dict[interest_pair[1]]:
                    if e2 in sent.text:
                        e2_appears = True
                if e1_appears and e2_appears: # if sentence has both entities
                    is_target_sent = True
                    sent_related_interest_pairs.append(interest_pair)
                    
            if is_target_sent: # sentence of interest
                for rip in sent_related_interest_pairs:
                    news_svo_info[rip].extend(findSVOs(sent))
                    nouns, verbs, adjs, advs, all_words = findGenerals(sent)
                    news_nn_info[rip].extend(nouns)
                    news_vb_info[rip].extend(verbs)
                    news_jj_info[rip].extend(adjs)
                    news_rb_info[rip].extend(advs)
                    news_all_info[rip].extend(all_words)
                    news_samples[rip].append(sent.text)
                    
    return news_index, news_svo_info, news_nn_info, news_vb_info, news_jj_info, news_rb_info,\
        news_all_info, news_samples
        

def process_news_file(filename):
    f = open(filename, 'r')

    head, tail = os.path.split(filename)
    year = tail[:2]
    #print(year)
    month = tail[3:5]
    #print(month)
    time = (int(year) - year_base) * 12 + int(month)
    short_filename = tail[:-4]

    news_list = f.readlines()
    entity_svo_dict = defaultdict(dict)
    entity_nn_dict = defaultdict(dict)
    entity_vb_dict = defaultdict(dict)
    entity_jj_dict = defaultdict(dict)
    entity_rb_dict = defaultdict(dict)
    entity_all_dict = defaultdict(dict)
    entity_sample_dict = defaultdict(dict)

    for news in news_list:
        related_pairs = find_related_entities(news, interest_pair_list)
        if len(related_pairs) > 0:
            news_index, news_svo_info, news_nn_info, news_vb_info, news_jj_info, news_rb_info,\
                news_all_info, news_samples = process_news_article(news, related_pairs)
            for k,v in news_svo_info.items():
                entity_svo_dict[k][(time, news_index)] = v
            for k,v in news_nn_info.items():
                entity_nn_dict[k][(time, news_index)] = v
            for k,v in news_vb_info.items():
                entity_vb_dict[k][(time, news_index)] = v
            for k,v in news_jj_info.items():
                entity_jj_dict[k][(time, news_index)] = v
            for k,v in news_rb_info.items():
                entity_rb_dict[k][(time, news_index)] = v
            for k,v in news_all_info.items():
                entity_all_dict[k][(time, news_index)] = v
            for k,v in news_samples.items():
                entity_sample_dict[k][(time, news_index)] = '\n'.join(v)
    
    f.close()
    pickle.dump(entity_svo_dict, open(write_path + 'large_svo_' + short_filename + '.pk', 'wb'))
    pickle.dump(entity_nn_dict, open(write_path + 'large_nn_' + short_filename + '.pk', 'wb'))
    pickle.dump(entity_vb_dict, open(write_path + 'large_vb_' + short_filename + '.pk', 'wb'))
    pickle.dump(entity_jj_dict, open(write_path + 'large_jj_' + short_filename + '.pk', 'wb'))
    pickle.dump(entity_rb_dict, open(write_path + 'large_rb_' + short_filename + '.pk', 'wb'))
    pickle.dump(entity_all_dict, open(write_path + 'large_all_' + short_filename + '.pk', 'wb'))
    pickle.dump(entity_sample_dict, open(write_path + 'large_sample_' + short_filename + '.pk', 'wb'))
    print(short_filename, 'finished')
    
    
filename_list = []
for filter_str in filter_strs:
    for filename in glob.glob(load_path + filter_str + '.txt'):
        filename_list.append(filename)

# TODO 
with Pool(40) as p: # fork 40 processes
    p.map(process_news_file, filename_list[0])
        
