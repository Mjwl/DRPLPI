# -*- coding: utf-8 -*-
###THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import catboost
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn import metrics
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import precision_recall_curve
import gzip
import pandas as pd
import pdb
import random
from random import randint
import scipy.io
from keras.models import Sequential
from keras.layers.core import  AutoEncoder, Dropout, Activation, Flatten, Merge, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import containers
from keras import regularizers
from keras.constraints import maxnorm
from attention import Attention,myFlatten

ALPHABET='ACGU'

def get_tris():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com   
           
def get_tri_nucleotide_composition(tris, seq):
    seq_len = len(seq)
    tri_feature = []
    for val in tris:
        num = seq.count(val)
        tri_feature.append(float(num)/seq_len)
    return tri_feature

def get_3_protein_trids():
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com

def get_3_protein_struct_trids():
    nucle_com = []
    chars = ['H', 'E', 'C']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com    

def translate_sequence (seq, TranslationDict):
    import string
    from_list = []
    to_list = []
    for k,v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)    
    TRANS_seq = seq.translate(string.maketrans(str(from_list), str(to_list)))    
    return TRANS_seq

def TransDict_from_list(groups):
    transDict = dict()
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group) 
        for c in g_members:            
            result[c] = str(tar_list[index]) 
        index = index + 1
    return result

def prepare_data(deepmind = False, seperate=False):
    print "loading data"
    lncRNA = pd.read_csv("zma_lncRNA.csv")
    protein = pd.read_csv("zma_rbp.csv")
    interaction = pd.read_fwf("ZMAInteraction.txt") 
    
    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    with open('ZMAInteraction.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label = values[1]
                name = values[0].split('_')
                protein = name[0] + '-' + name[1]
                RNA = name[0] + '-' + name[1]
                if label == 'interactive':
                    interaction_pair[(protein, RNA)] = 1
                else:
                    interaction_pair[(protein, RNA)] = 0
                index  = 0
            else:
                seq = line[:-1]
                if index == 0:
                    protein_seq_dict[protein] = seq
                else:
                    RNA_seq_dict[RNA] = seq
                index = index + 1
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()
    train = []
    label = []
    chem_fea = []
    feature = []
    for key, val in interaction_pair.iteritems():
        protein, RNA = key[0], key[1]
        
        if RNA_seq_dict.has_key(RNA) and protein_seq_dict.has_key(protein): 
            label.append(val)
            RNA_seq = RNA_seq_dict[RNA]
            protein_seq = translate_sequence (protein_seq_dict[protein], group_dict)
            if deepmind:
                RNA_tri_fea = rna_feature_extract(feature, RNA_seq)
                protein_tri_fea = protein_feature_extract(feature, protein_seq_dict[protein])
                train.append((RNA_tri_fea, protein_tri_fea))
            else:
                
                RNA_tri_fea = test_feature_extract(feature, RNA_seq)
      	        protein_tri_fea = test_protein_feature_extract(feature, protein_seq_dict[protein])
                if seperate:
                    tmp_fea = (protein_tri_fea, RNA_tri_fea)
                    
                else:
                    tmp_fea = protein_tri_fea + RNA_tri_fea                    
                train.append(tmp_fea)
                
        else:
            print RNA, protein   
    
    return np.array(train), label

def SSEC():
    label = []
    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    protein_index = 1
    with open('zma_rbpStruct.fa', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip()
                label_temp = values[1]
                protein = values[0]
                if label_temp=='1':
                    label.append(1)
                else:
                    label.append(0)
            else:
                seq = line[:-1]
                protein_seq_dict[protein_index] = seq
                protein_index = protein_index + 1
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE','C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_struct_trids()
    # tris3 = get_3_trids()
    bpf = []
    kmer = []
    for i in protein_seq_dict:  

        protein_seq = translate_sequence(protein_seq_dict[i], group_dict)
        protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)
        kmer.append(protein_tri_fea)
        protein_index = protein_index + 1
        # chem_fea.append(chem_tmp_fea)

    return np.array(kmer[0])

def kmer(kmerid, k):
	kmer = ''
	nts = ['A', 'C', 'G', 'T']
	for i in xrange(k):
		kmer = nts[(kmerid % 4)] + kmer
		kmerid = int(kmerid/4)

	return kmer

def InsertgappedVect(seq, g=8):  # 1___1
    feature=[]
    m = list(itertools.product(ALPHABET, repeat=2))
    for i in range(1, g + 1, 1):
        V = kmer(len(seq), i + 2)
        for gGap in m:
            count = 0
            for v in V:
                if v[0] == gGap[0] and v[-1] == gGap[1]:
                    count += 1
            feature.append(count)
    return feature

def reverse_complement_features(feature, seq):
    reverse_complements_4 = ['CGCC', 'CTAA', 'GGCG', 'TTAG']
    reverse_complements_5 = ['AAAAG', 'CTTTT', 'AGATA', 'TATCT', 'CCCAC', 'GTGGG', 'CGCAC', 'GTGCG', 'CTAAG', 'CTTAG', 'GGCAC', 'GTGCC', 'GGCCA', 'TGGCC', 'TATAA', 'TTATA', 'TATCA', 'TGATA', 'TATGA', 'TCATA']
    L4 = len(seq) - 4 + 1
    L5 = len(seq) - 5 + 1
    
    count_1 = 0
    count_2 = 0

    for i in range(0, len(seq) - 4 + 1):
    	pattern = []
    	for j in range(0, 4):
    		pattern.append(seq[i+j])
    	s = ''.join(pattern)
    	
    	if(s in reverse_complements_4):
    		if(s == 'CGCC' or s == 'CTAA'):
    			count_1 += 1
    		else:
    			count_2 += 1
    feature.append(count_1/L4)
    feature.append(count_2/L4)

    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_6 = 0
    count_7 = 0
    count_8 = 0

    for i in range(0, len(seq) - 5 + 1):
    	pattern = []
    	for j in range(0, 5):
    		pattern.append(seq[i+j])
    	s = ''.join(pattern)
    	
    	if(s in reverse_complements_5):
    		if(s == 'AAAAG' or s == 'CTTTT'):
    			count_1 += 1
    		elif(s == 'AGATA' or s == 'TATCT'):
    			count_2 += 1
    		elif(s == 'CCCAC' or s == 'GTGGG'):
    			count_3 += 1
    		elif(s == 'CGCAC' or s == 'GTGCG'):
    			count_4 += 1
    		elif(s == 'CTAAG' or s == 'CTTAG'):
    			count_5 += 1
    		elif(s == 'GGCAC' or s == 'GTGCC'):
    			count_6 += 1
    		elif(s == 'GGCCA' or s == 'TGGCC'):
    			count_7 += 1
    		elif(s == 'TATAA' or s == 'TTATA'):
    			count_8 += 1

    feature.append(count_1/L5)
    feature.append(count_2/L5)
    feature.append(count_3/L5)
    feature.append(count_4/L5)
    feature.append(count_5/L5)
    feature.append(count_6/L5)
    feature.append(count_7/L5)
    feature.append(count_8/L5)
       
    return feature

finstr = open('zma_RNAStruct.fa', "r")
t = {}
bar = 5

res_list = []

def rna_feature_extract(feature, seq):
	tris = get_tris()
	feature = []
	tkmer = get_tri_nucleotide_composition(tris, seq)
        gkm = InsertgappedVect(seq, g=8)	
	rcf = reverse_complement_features(feature, seq)        
	pred_list = []

	index = -1 
	for index_1, line in enumerate(finstr):

    	    if line[0] == '>':
                continue
            index += 1

            if index % 101 == 0 and index != 0:
               num_list = []
               list = []
               for (k, v) in t.items():
                   num_list.append(v)
                   num_list.sort(reverse=True)
                   res = 0
                   total_num = 0
        	   sum = 0
        	   base_line = 0
        	   if len(num_list) > bar:
            	      base_line = num_list[bar]
                   for i in range(min(bar, len(num_list))):
                       sum += num_list[i]
               if num_list[i] > base_line:
                   total_num += 1
                   res_list.append(total_num)
                   last_seq = " "
               for (k, v) in t.items():
                  if v > base_line or (v == base_line and total_num < 5):
                      last_seq = k
                      list.append(1. * v / 100)
                      if len(num_list) > bar and v == num_list[bar]:
                          total_num += 1

               pred = 1. * list[len(list) - 1] / (5 - total_num + 1)
               list[len(list) - 1] = pred
               for i in range(total_num, 5):
                    list.append(pred)

               list.append(1. * (100 - sum) / 100)
               pred_list.append(np.array(list))
               t.clear()

            if index % 101 == 0:
               continue

            a = line.split(" ")
            if a[0] in t:
               t[a[0]] = t[a[0]] + 1
            else:
                t[a[0]] = 1

        num_list = []
        list = []
        for (k, v) in t.items():
            num_list.append(v)
        num_list.sort(reverse=True)
  	res = 0
	total_num = 0
	sum = 0
	base_line = 0
	if len(num_list) > bar:
    	    base_line = num_list[bar]
	for i in range(min(bar, len(num_list))):
    	    sum += num_list[i]
    	    if num_list[i] > base_line:
               total_num += 1
        res_list.append(total_num)
	last_seq = " "
	for (k, v) in t.items():
    	    if v > base_line or (v == base_line and total_num < 5):
               last_seq = k
               list.append(1. * v / 100)
               if len(num_list) > bar and v == num_list[bar]:
                  total_num += 1

        pred = 1. / 5 
        for i in range(total_num, 5):
            list.append(pred)
            list.append(1. * (100 - sum) / 100)

	pred_list.append(np.array(list))
	t.clear()
	pred_list = np.array(pred_list)

	pred_list = np.array(list)

	result = pred_list

	kmer = gkm+tkmer+rcf
                                                     
        feature = np.append(kmer,result[:5])
	return feature

def BPF(seq_temp):
    seq = seq_temp
    chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    fea = []
    tem_vec =[]
    k = 16
    for i in range(k):
        if seq[i] =='A':
            tem_vec = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='C':
            tem_vec = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='D':
            tem_vec = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='E':
            tem_vec = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='F':
            tem_vec = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='G':
            tem_vec = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='H':
            tem_vec = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='I':
            tem_vec = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='K':
            tem_vec = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='L':
            tem_vec = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='M':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif seq[i]=='N':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif seq[i]=='P':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif seq[i]=='Q':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif seq[i]=='R':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif seq[i]=='S':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif seq[i]=='T':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif seq[i]=='V':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif seq[i]=='W':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif seq[i]=='Y':
            tem_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        fea = fea + tem_vec
    return fea

def protein_feature_extract(feature, seq):	
	tris = get_tris()
	feature = []	
	bpf = BPF(seq)
	sse = SSEC()
        feature = np.append(sse,bpf)
	return feature

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
    else:
        precision = float(tp)/(tp+ fp)
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, precision, sensitivity, specificity, MCC 

def read_fasta_file(fasta_file):
    seq_dict = {}    
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[1:].upper() #discarding the initial >
            seq_dict[name] = ''
        else:
            #it is sequence
            seq_dict[name] = seq_dict[name] + line
    fp.close()
    
    return seq_dict

def read_name_from_fasta(fasta_file):
    name_list = []
    fp = open(fasta_file, 'r')
    for line in fp:
        if line[0] == '>':
            name = line.rstrip('\r\n')[1:]
            name_list.append(name.upper())
    fp.close()
    return name_list

def transfer_array_format(data):
    formated_matrix1 = []
    formated_matrix2 = []
    for val in data:
        formated_matrix1.append(val[0])
        formated_matrix2.append(val[1])    
    return np.array(formated_matrix1), np.array(formated_matrix2)

def preprocess_data(X, scaler=None, stand = True):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def get_blend_data(j, clf, skf, X_test, X_dev, Y_dev, blend_train, blend_test):
        print 'Training classifier [%s]' % (j)
        blend_test_j = np.zeros((X_test.shape[0], len(skf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):
            print 'Fold [%s]' % (i)
            
            # This is the training and validation set
            X_train = X_dev[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev[cv_index]
            Y_cv = Y_dev[cv_index]
            
            clf.fit(X_train, Y_train)
            blend_train[cv_index, j] = clf.predict_proba(X_cv)[:,1]
            blend_test_j[:, i] = clf.predict_proba(X_test)[:,1]
        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)    
        print 'Y_dev.shape = %s' % (Y_dev.shape)
     
def multiple_layer_autoencoder(X_train, X_test, activation = 'linear', batch_size = 100, nb_epoch = 20, last_dim = 64):
    nb_hidden_layers = [X_train.shape[1], 256, 128, last_dim]
    X_train_tmp = np.copy(X_train)
    #X_test_tmp = np.copy(X_test)
    encoders = []
    for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
        print('Training the layer {}: Input {} -> Output {}'.format(i, n_in, n_out))
        # Create AE and training
        ae = Sequential()
	num_hidden = 128
	max_features = X_train.shape[1]
	encoder = Embedding(max_features, 256)
	encoder = LSTM(256, num_hidden, activation='sigmoid', inner_activation='hard_sigmoid')
	encoder = Dense(n_in, n_out, activation=activation, init='glorot_normal', W_constraint = unitnorm())
		
	decoder = Attention(max_features, 256) 
	decoder = LSTM(256, num_hidden, activation='sigmoid', inner_activation='hard_sigmoid')     
	decoder = Dense(n_out, n_in, activation=activation, init='glorot_normal'), W_constraint = unitnorm())
        ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                           output_reconstruction=False))
        ae.add(Dropout(0.5))
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
        ae.compile(loss='mean_squared_error', optimizer=adadelta)#'rmsprop')
        ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, callbacks = [EarlyStopping(monitor='val_acc', patience=2)])
        encoders.append(ae)
        X_train_tmp = ae.predict(X_train_tmp)
        print X_train_tmp.shape
        #X_test_tmp = ae.predict(X_test_tmp)
        
    #return encoders, X_train_tmp, X_test_tmp
    return encoders      

def autoencoder_two_subnetwork_fine_tuning(X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = None, batch_size =100, nb_epoch = 20):
    print 'autoencode learning'
    last_dim = 64
    encoders1 = multiple_layer_autoencoder(X_train1, X_test1, activation = 'sigmoid', batch_size = batch_size, nb_epoch = nb_epoch, last_dim = last_dim)
    encoders2 = multiple_layer_autoencoder(X_train2, X_test2, activation = 'sigmoid', batch_size = batch_size, nb_epoch = nb_epoch, last_dim = last_dim)
    #pdb.set_trace()
    X_train1_tmp_bef = np.copy(X_train1)
    X_test1_tmp_bef = np.copy(X_test1) 
    for ae in encoders1:
        X_train1_tmp_bef = ae.predict(X_train1_tmp_bef)
        print X_train1_tmp_bef.shape
        X_test1_tmp_bef = ae.predict(X_test1_tmp_bef)    
    X_train2_tmp_bef = np.copy(X_train2)
    X_test2_tmp_bef = np.copy(X_test2) 
    for ae in encoders2:
        X_train2_tmp_bef = ae.predict(X_train2_tmp_bef)
        print X_train2_tmp_bef.shape
        X_test2_tmp_bef = ae.predict(X_test2_tmp_bef)
        
    prefilter_train_bef = np.concatenate((X_train1_tmp_bef, X_train2_tmp_bef), axis = 1)
    prefilter_test_bef = np.concatenate((X_test1_tmp_bef, X_test2_tmp_bef), axis = 1)
        
    print 'fine tunning'
    print 'number of layers:', len(encoders1)
    sec_num_hidden = last_dim
    model1 = Sequential()
    ind = 0
    for encoder in encoders1:
        model1.add(encoder.layers[0].encoder)
        if ind != len(encoders1)  - 1 :
            model1.add(Dropout(0.5)) 
            ind = ind + 1
    model1.add(PReLU((sec_num_hidden,)))
    model1.add(BatchNormalization((sec_num_hidden,)))
    model1.add(Dropout(0.5))    

    model2 = Sequential()
    ind = 0
    for encoder in encoders2:
        model2.add(encoder.layers[0].encoder)
        if ind != len(encoders2)  - 1 :
            model2.add(Dropout(0.5)) 
            ind = ind + 1
    model2.add(PReLU((sec_num_hidden,)))
    model2.add(BatchNormalization((sec_num_hidden,)))   
    model2.add(Dropout(0.5))     
         
    model = Sequential()
    model.add(Merge([model1, model2], mode='concat'))
    total_hid = sec_num_hidden + sec_num_hidden
    
    model.add(Dense(total_hid, 2))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))
    #sgd = SGD(lr=3, decay=1e-6, momentum=0.9, nesterov=True)
    adadelta = Adadelta(lr=3.0, rho=0.95, epsilon=1e-08)
    model.compile(loss='hinge', optimizer=adadelta) #'rmsprop')
    model.fit([X_train1, X_train2], Y_train, batch_size=100, nb_epoch=30, verbose=0, callbacks = [EarlyStopping(monitor='val_acc', patience=2)])
    X_train1_tmp = np.copy(X_train1)
    X_test1_tmp = np.copy(X_test1)  
    ae=model.layers[0].layers[0]  
    ae.compile(loss='mean_squared_error', optimizer='adam')
    X_train1_tmp = ae.predict(X_train1_tmp)
    X_test1_tmp = ae.predict(X_test1_tmp)

    X_train2_tmp = np.copy(X_train2)
    X_test2_tmp = np.copy(X_test2)  
    ae=model.layers[0].layers[1]  
    ae.compile(loss='mean_squared_error', optimizer='adam')
    X_train2_tmp = ae.predict(X_train2_tmp)
    X_test2_tmp = ae.predict(X_test2_tmp)
    
    prefilter_train = np.concatenate((X_train1_tmp, X_train2_tmp), axis = 1)
    prefilter_test = np.concatenate((X_test1_tmp, X_test2_tmp), axis = 1)
    return prefilter_train, prefilter_test, prefilter_train_bef, prefilter_test_bef
    #return model
def plot_roc_curve(labels, probality, legend_text, auc_tag = True):
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text +' (AUC=%6.3f) ' %roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text )

def plot_auprc_curve(labels, probality, legend_text, auprc_tag = True):
    precision1, recall, threshods = precision_recall_curve(labels, probality)
    aupr_score = auc(recall, precision1)
    if auprc_tag:
        rects1 = plt.plot(recall, precision1, label=legend_text +' (AUPRC=%6.3f) ' %aupr_score)
    else:
        rects1 = plt.plot(recall, precision1, label=legend_text )

def DeepRPLPI():
    X, labels = prepare_data(seperate = True)
    #X, labels = get_data_deepmind(dataset='NPInter', seperate=True)
    
    X_data1, X_data2 = transfer_array_format(X)
    print X_data1.shape, X_data2.shape
    X_data1, scaler1 = preprocess_data(X_data1)
    X_data2, scaler2 = preprocess_data(X_data2)
    y, encoder = preprocess_labels(labels)
    
    num_cross_val = 5
    all_performance = []
    all_performance_rf = []
    all_performance_bef = []
    all_performance_xt = []
    all_performance_dt = []
    all_performance_gb = []
    all_performance_xgb = []
    all_performance_ctb = []
    all_performance_ensemb = []    
    all_performance_blend = []
    all_labels = []
    all_prob = {}
    num_classifier = 3
    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []
    all_prob[3] = []
    all_prob[4] = []
    all_prob[5] = []
    all_prob[6] = []
    all_prob[7] = []
    all_prob[8] = []
    for fold in range(num_cross_val):
        train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])  
          
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        
        blend_train = np.zeros((train1.shape[0], num_classifier)) # Number of training data x Number of classifiers
        blend_test = np.zeros((test1.shape[0], num_classifier)) # Number of testing data x Number of classifiers 
        skf = list(StratifiedKFold(train_label_new, num_classifier))  
        class_index = 0
        prefilter_train, prefilter_test, prefilter_train_bef, prefilter_test_bef = autoencoder_two_subnetwork_fine_tuning(train1, train2, train_label, test1, test2, test_label)
        
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
                
        all_labels = all_labels + real_labels
        
        '''
        prefilter_train1 = xgb.DMatrix( prefilter_train, label=train_label_new)
        evallist  = [(prefilter_train1, 'train')]
        num_round = 10
        clf = xgb.train( plst, prefilter_train1, num_round, evallist )
        prefilter_test1 = xgb.DMatrix( prefilter_test)
        ae_y_pred_prob = clf.predict(prefilter_test1)
        '''
        tmp_aver = [0] * len(real_labels)
        print 'deep autoencoder'
        clf = RandomForestClassifier(n_estimators=50)
	clf.fit(prefilter_train, train_label_new)
        ae_y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        tmp_aver = [val1 + val2/3 for val1, val2 in zip(ae_y_pred_prob, tmp_aver)]
        proba = transfer_label_from_prob(ae_y_pred_prob)         
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
	auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
        all_performance.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
                
        print 'deep autoencoder without fine tunning'
        class_index = class_index + 1
        clf = RandomForestClassifier(n_estimators=50) 
        clf.fit(prefilter_train, train_label_new)
        ae_y_pred_prob_bef = clf.predict_proba(prefilter_test_bef)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob_bef]
        tmp_aver = [val1 + val2/3 for val1, val2 in zip(ae_y_pred_prob_bef, tmp_aver)]
        proba = transfer_label_from_prob(ae_y_pred_prob_bef)       
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
	precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
        all_performance_bef.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])

        print 'Random Forest'
	class_index = class_index + 1
	clf = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
        clf.fit(prefilter_train, train_label_new)
        ae_y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        proba = transfer_label_from_prob(ae_y_pred_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print "RF :", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
	all_performance_rf.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
        
        print 'Extra trees'
	class_index = class_index + 1
	etree = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,max_features=50, max_depth=40, min_samples_leaf=4)
        etree.fit(prefilter_train, train_label_new)
        ae_y_pred_prob = etree.predict_proba(prefilter_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        proba = transfer_label_from_prob(ae_y_pred_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print "Extra-trees :", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
	all_performance_xt.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
        
        print 'Decision Tree'
	class_index = class_index + 1
	clf = DecisionTreeClassifier()
        clf.fit(prefilter_train, train_label_new)
        ae_y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        proba = transfer_label_from_prob(ae_y_pred_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print "dt :", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
	all_performance_dt.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
        	
	print 'Gradient boosting'
	class_index = class_index + 1
        clf = GradientBoostingClassifier(n_estimators=70, random_state=7)
        clf.fit(prefilter_train, train_label_new)
        gb_ae_y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in gb_ae_y_pred_prob]
        gb_proba = transfer_label_from_prob(gb_ae_y_pred_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), gb_proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, gb_ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        gb_precision1, gb_recall, pr_threshods = precision_recall_curve(real_labels, gb_ae_y_pred_prob)
        gb_aupr_score = auc(gb_recall, gb_precision1)
        print "GB :", acc, precision, sensitivity, specificity, MCC, auc_score, gb_aupr_score
	all_performance_gb.append([acc, precision, sensitivity, specificity, MCC, auc_score, gb_aupr_score])
        
	print 'Extreme Gradient boosting'
	class_index = class_index + 1
        clf = XGBClassifier()
        clf.fit(prefilter_train, train_label_new)
        ae_y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        proba = transfer_label_from_prob(ae_y_pred_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)     
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print "XGB :", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
	all_performance_xgb.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])           

	print 'CatBoost'
	class_index = class_index + 1
        prefilter_train = np.concatenate((train1, train2), axis = 1)
        prefilter_test = np.concatenate((test1, test2), axis = 1)
        cmodel = catboost.CatBoostClassifier(iterations=150,depth=15,learning_rate=0.5,loss_function='Logloss', verbose=False)
	print ('begin to predict data')
	cmodel.fit(prefilter_train, train_label_new)
	pred_period = cmodel.predict(prefilter_test)
        all_prob[class_index] = all_prob[class_index] + [val for val in pred_period]
	proba = transfer_label_from_prob(pred_period)
	acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, pred_period)
        auc_score = auc(fpr, tpr)
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, pred_period)
        aupr_score = auc(recall, precision1)
        print "CatBoost :", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
	all_performance_ctb.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
	
	print 'Ensemble Classifiers'
	class_index = class_index + 1
        prefilter_train = np.concatenate((train1, train2), axis = 1)
        prefilter_test = np.concatenate((test1, test2), axis = 1)
	etree = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,max_features=50, max_depth=40, min_samples_leaf=4)
	cmodel = catboost.CatBoostClassifier(iterations=150,depth=15,learning_rate=0.5,loss_function='Logloss', verbose=False) 
	eclf = VotingClassifier(estimators=[('ctb', cmodel), ('xt', etree)], voting='soft')
	eclf.fit(prefilter_train, train_label_new)
	ae_y_pred_prob = eclf.predict_proba(prefilter_test)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        proba = transfer_label_from_prob(ae_y_pred_prob)        
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
	precision1, recall, threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        print "Ens :", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
        all_performance_ensemb.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score]) 
        print '---' * 50               
    
    print 'mean performance of random forest using raw feature'
    print np.mean(np.array(all_performance_rf), axis=0)
    print '---' * 50   
    print 'mean performance of Extra trees using raw feature'
    print np.mean(np.array(all_performance_xt), axis=0)
    print '---' * 50   
    print 'mean performance of Decision Tree using raw feature'
    print np.mean(np.array(all_performance_dt), axis=0)
    print '---' * 50 
    print 'mean performance of Gradient Boosting using raw feature'
    print np.mean(np.array(all_performance_gb), axis=0)
    print '---' * 50 
    print 'mean performance of Extreme Gradient Boosting using raw feature'
    print np.mean(np.array(all_performance_xgb), axis=0)
    print '---' * 50  
    print 'mean performance of CatBoosting using raw feature'
    print np.mean(np.array(all_performance_ctb), axis=0)
    print '---' * 50  
    print 'mean performance of Classifier Ensemble using raw feature'
    print np.mean(np.array(all_performance_ensemb), axis=0)
    print '---' * 50   
    print 'mean performance of stacked ensembling'
    print np.mean(np.array(all_performance_blend), axis=0)
    print '---' * 50
   
    Figure = plt.figure()
    plot_auprc_curve(all_labels, all_prob[8], 'DRPLPI')
    plot_auprc_curve(all_labels, all_prob[7], 'CatBoost')
    plot_auprc_curve(all_labels, all_prob[3], 'Extra Trees')
    plot_auprc_curve(all_labels, all_prob[2], 'XGB')
    plot_auprc_curve(all_labels, all_prob[6], 'RF') 
    plot_auprc_curve(all_labels, all_prob[4], 'DT')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPRC')
    plt.legend(loc="lower right")
    plt.show()    

def transfer_label_from_prob(proba):
    label = [1 if val>=0.5 else 0 for val in proba]
    return label

if __name__=="__main__":
    DeepRPLPI()
