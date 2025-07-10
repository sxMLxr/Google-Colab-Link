#!/usr/bin/env python
# coding: utf-8

# # Week 3 - Session 2: Recent Temporal Pattern (RTP) Mining

# In[1]:

import pandas as pd
import numpy as np
import pickle
import sys
import random
from sklearn.metrics import recall_score, roc_curve, auc, accuracy_score, confusion_matrix, precision_score, f1_score

import TemporalAbstraction
import RTPmining
import classifier
from Config import Options
import time


# RTP Mining 
# C1: positive class , C0: negative class 
def store_patterns(i, trainC1, trainC0, opts):

    # -------------------------------
    # RTP mining for each class
    C1_patterns = RTPmining.pattern_mining(trainC1, opts.max_g, opts.sup_pos*len(trainC1), opts)
    print("Total # patterns from positive:", len(C1_patterns))
    C0_patterns = RTPmining.pattern_mining(trainC0, opts.max_g, opts.sup_neg*len(trainC0), opts)
    print("Total # patterns from negative:", len(C0_patterns))

    ############## Writing patterns to the files #################
    C1_pos_file = open(opts.patterns_path + 'C1_pos_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.txt','w')
    C0_neg_file = open(opts.patterns_path + 'C0_neg_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.txt','w')
    for p in C1_patterns:
        C1_pos_file.write(p.describe())
    for p in C0_patterns:
        C0_neg_file.write(p.describe())

    pos_fname = opts.patterns_path + 'C1_pos_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.pckl'
    neg_fname = opts.patterns_path + 'C0_neg_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.pckl'

    f = open(pos_fname, 'wb')
    pickle.dump(C1_patterns, f)
    f.close()
    f = open(neg_fname, 'wb')
    pickle.dump(C0_patterns, f)
    f.close()

    return C1_patterns, C0_patterns

def load_patterns(i, opts):
    pos_fname = opts.patterns_path + 'C1_pos_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.pckl'
    neg_fname = opts.patterns_path + 'C0_neg_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.pckl'
    f = open(pos_fname, 'rb')
    C1_patterns = pickle.load(f)
    f.close()
    f = open(neg_fname, 'rb')
    C0_patterns = pickle.load(f)
    f.close()
    return C1_patterns, C0_patterns

def random_subset(iterator, K):
    result = []
    N = 0
    for item in iterator:
        N += 1
        if len( result ) < K:
            result.append( item )
        else:
            s = int(random.random() * N)
            if s < K:
                result[ s ] = item
    return result


def make_MSS(pos_events, neg_events, opts):
    # -------------------------------------------------------    
    # 0. Data Preprocessiong
    # -------------------------------------------------------    
    # 0-1. Data alignment 
    # Right aligned: a backward time window from the shock onset time
    if opts.early_prediction > 0 and opts.alignment == 'right':
        pos_cut = pos_events[pos_events.EventTime - pos_events[opts.timestamp_variable] >= opts.early_prediction * 60].copy(deep=True)
        neg_cut = neg_events[neg_events.LastMinute - neg_events[opts.timestamp_variable] >= opts.early_prediction * 60].copy(deep=True)
        if opts.observation_window:
            pos_cut = pos_cut[pos_cut.EventTime - pos_cut[opts.timestamp_variable] <= 60 * (opts.observation_window + opts.early_prediction)]
            neg_cut = neg_cut[neg_cut.LastMinute - neg_cut[opts.timestamp_variable] <= 60 * (opts.observation_window + opts.early_prediction)]
            
    # Left aligned: a time window starting from the begning of the trajectory
    elif opts.observation_window and opts.alignment == 'left':
        pos_cut = pos_events[pos_events[opts.timestamp_variable] <= opts.early_prediction * 60]
        neg_cut = neg_events[neg_events[opts.timestamp_variable] <= opts.early_prediction * 60]
    
    # When using 'trunc' mode, we use truncated data for both training and test     
    if opts.settings == 'trunc':
        pos_events = pos_cut.copy(deep=True)
        neg_events = neg_cut.copy(deep=True)

    # 0-2. Sampling for balanced data (because we may lose some samples by data alignment)      
    # Sampling from negative visits, if # of negative trajectories > # of positive trajectories        
    if len(neg_events[opts.unique_id_variable].unique()) > len(pos_events[opts.unique_id_variable].unique()):
        neg_id = random_subset(neg_events[opts.unique_id_variable].unique(), len(pos_events[opts.unique_id_variable].unique()))
        neg_events = neg_events[neg_events[opts.unique_id_variable].isin(neg_id)]
        
    # Sampling from positive visits, if # of positives > # of negatives        
    if len(pos_events[opts.unique_id_variable].unique()) > len(neg_events[opts.unique_id_variable].unique()):
        pos_id = random_subset(pos_events[opts.unique_id_variable].unique(), len(neg_events[opts.unique_id_variable].unique()))
        pos_events = pos_events[pos_events[opts.unique_id_variable].isin(pos_id)]

    # -------------------------------------------------------    
    #  1. Temporal Abstraction
    # -------------------------------------------------------    
    #  1-1. Discretize numerical values and abstract temporal states over time (please see the lecture 6)
    for f in opts.numerical_feat:
        pos_events.loc[:,f], neg_events.loc[:,f] = TemporalAbstraction.abstraction_alphabet(pos_events[f], neg_events[f])
    
    # When using 'entire' mode, we need truncated data for test.     
    if opts.settings == 'entire':    
        for f in opts.numerical_feat:
            pos_cut.loc[:,f], neg_cut.loc[:,f] = TemporalAbstraction.abstraction_alphabet(pos_cut[f], neg_cut[f])

    
    # -------------------------------------------------------    
    #  1-2. Make Multivariate State Sequence (MSS), consisting of State Intervals
    #  State Interval = (feature, value, start, end)

    #  MSS for positive cases
    MSS_positive = []
    grouped = pos_events.groupby(opts.unique_id_variable)
    for name, group in grouped:
        group = group.sort_values([opts.timestamp_variable])
        MSS_positive.append(TemporalAbstraction.MultivariateStateSequence(group, opts))
    #print(len(MSS_positive))

    #  MSS for negative cases
    MSS_negative = []
    grouped = neg_events.groupby(opts.unique_id_variable)
    for name, group in grouped:
        group = group.sort_values([opts.timestamp_variable])
        MSS_negative.append(TemporalAbstraction.MultivariateStateSequence(group, opts))
    #print(len(MSS_negative))
    
    # Save the generated MSSs for each class (to skip this long process when comparing different classifiers)
    f = open('RTP_log/MSS_pos_'+opts.alignment+'_'+str(opts.early_prediction)+'.pckl', 'wb')
    pickle.dump(MSS_positive, f)
    f.close()
    f = open('RTP_log/MSS_neg_'+opts.alignment+'_'+str(opts.early_prediction)+'.pckl', 'wb')
    pickle.dump(MSS_negative, f)
    f.close()

    return MSS_positive, MSS_negative


def load_MSS(opts):
    f = open('RTP_log/MSS_pos_'+opts.alignment+'_'+str(opts.early_prediction)+'.pckl', 'rb')
    MSS_positive = pickle.load(f)
    f.close()
    f = open('RTP_log/MSS_neg_'+opts.alignment+'_'+str(opts.early_prediction)+'.pckl', 'rb')
    MSS_negative = pickle.load(f)
    f.close()

    return MSS_positive, MSS_negative


# k-fold cross validataion (the k-fold is definece in Config.py)

def pred_fold(MSS_positive, MSS_negative, opts):
    
    
    if opts.alignment == 'right':
        if len(MSS_negative) > len(MSS_positive):
            MSS_negative = random_subset(MSS_negative, len(MSS_positive))

    print ("size of negative data:", len(MSS_negative))
    print ("size of positive data:", len(MSS_positive))

    C0_subset_size = len(MSS_negative)//opts.num_folds # C0 = negative class
    C1_subset_size = len(MSS_positive)//opts.num_folds # C1 = positive class

    test_pred, test_pred_prob, train_pred, test_labels = [], [],[], []
    
    for i in range(opts.num_folds):
        print( "***************** FOLD ", i+1, "*****************")      
        trainC0 = MSS_negative[:i*C0_subset_size] + MSS_negative[(i+1)*C0_subset_size:]
        trainC1 = MSS_positive[:i*C1_subset_size] + MSS_positive[(i+1)*C1_subset_size:]
        
        if opts.settings == 'trunc':
            testC0 = MSS_negative[i*C0_subset_size:][:C0_subset_size]
            testC1 = MSS_positive[i*C1_subset_size:][:C1_subset_size]
        
        print ("Size of positive and negative training:", len(trainC1), len(trainC0))
        print( "Size of positive and negative test:", len(testC1), len(testC0))       
        
        # ------------------------------------------------
        # 2. RTP (Recent Temporal Pattern) mining
        
        # ---- either store patterns or load the dumped ones
        C1_patterns, C0_patterns = store_patterns(i, trainC1, trainC0, opts)
        # C1_patterns, C0_patterns = load_patterns(i, opts)
        # ------------------------------------------------
        
        # Get all the unique patterns
        all_patterns = list(C1_patterns)
        for j in range(0,len(C0_patterns)):
            if not any((x == C0_patterns[j]) for x in all_patterns):
                all_patterns.append(C0_patterns[j])
        print ("number of all patterns:", len(all_patterns))

        # Make the training and test set
        train_data = list(trainC1)
        train_data.extend(trainC0)
        train_labels = list(np.ones(len(trainC1)))
        train_labels.extend(np.zeros(len(trainC0)))

        test_data = list(testC1)
        test_data.extend(testC0)
        test_labels = list(np.ones(len(testC1)))
        test_labels.extend(np.zeros(len(testC0)))
        
        # 3. Data Transformation
        train_binary_matrix = pd.DataFrame(classifier.create_binary_matrix(train_data, all_patterns, opts.max_g))
        test_binary_matrix = pd.DataFrame(classifier.create_binary_matrix(test_data, all_patterns, opts.max_g))

        # 4. Classification #
        if opts.classifier == 'svm':
            print("--------", opts.classifier)
            trp, tsp, tsp_prob = classifier.learn_classifier(opts, train_binary_matrix, train_labels, test_binary_matrix, test_labels)
        else:
            print("--------", opts.classifier)
            trp, tsp, tsp_prob = classifier.learn_logistic_regression(train_binary_matrix, train_labels, test_binary_matrix, test_labels)
        
        print("--x------", opts.classifier)
            
        print(len(tsp))
        test_labels.extend(test_labels)
        test_pred.extend(tsp)
        train_pred.extend(trp)
        for each in tsp_prob:
            test_pred_prob.append(each[1])
        print(len(tsp_prob))

    print(confusion_matrix(test_labels, test_pred))
    accuracy = accuracy_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred)
    recall = recall_score(test_labels, test_pred)
    f_measure = f1_score(test_labels, test_pred)
    fpr, tpr, _ = roc_curve(test_labels, test_pred_prob)
    roc_auc = auc(fpr, tpr)
    print("Results: ", accuracy, precision, recall, f_measures, roc_auc)
    return accuracy, precision, recall, f_measure, roc_auc



# According to the recommendation of medical experts: 
# Forward filling for 8 hours for vital signs and 24 hours for lab results
def expertImputation(opts):
    ndf_ni = pd.read_csv('mimic/mimic_nonshock_noImpute.csv',header=0)
    sdf_ni = pd.read_csv('mimic/mimic_shock_noImpute.csv',header=0)
    # ---------------------------------------------------------------------------
    # Assume that we are given the mean values from the training set as follows:    
    df = pd.concat([ndf_ni, sdf_ni], axis=0, sort=False)
    meanTrain = df.groupby(opts.unique_id_variable)[opts.all_feat].mean().mean().values
    # ---------------------------------------------------------------------------    
    
    for f in opts.numerical_feat: 
        if f in opts.vitals or f in opts.oxygenCtrl:
            ffill_window = 8 * 60
        elif f in opts.labs:
            ffill_window = 24 * 60
            
        ndf = carryFwd_minutes(opts, ndf_ni, f, ffill_window)
        sdf = carryFwd_minutes(opts, sdf_ni, f, ffill_window)

    # Mean imputation
    sdf = meanImpute(sdf, opts.all_feat, meanTrain)
    ndf = meanImpute(ndf, opts.all_feat, meanTrain)
        
    ndf.to_csv('mimic/mimic_nonshock.csv', index=False)
    sdf.to_csv('mimic/mimic_shock.csv', index=False)
    del ndf_ni, sdf_ni, ndf, sdf

def carryFwd_minutes(opts, df, feat, cftime):

    # Get the observed time
    df.loc[pd.notnull(df[feat]),'observedTime'] = df.loc[:,opts.timestamp_variable]
    
    # Forward filling with observed values
    df.loc[:,feat] = df.groupby(opts.unique_id_variable)[feat].ffill()
    # Forward filling with observed time
    df.loc[:,'observedTime'] = df.groupby(opts.unique_id_variable)['observedTime'].ffill()
    # Set NA for the imputed values out of the given time frame
    df.loc[df[opts.timestamp_variable]- df['observedTime'] > cftime, feat] = np.nan    
    # ---------------------------------------------------------   
    
    df = df.drop('observedTime', axis = 1)
    
    return df
    
# imputation with training data 
def meanImpute(df, feat, meanTrain):
    for i in range(len(feat)):
        df[feat[i]] = df[feat[i]].fillna(meanTrain[i])
    return df

    
# Main function
def main():
    start = time.time()
    
    opts = Options()
    print("feat:", opts.numerical_feat)

    expertImputation(opts)   # Added : Missing data handling 
    
    pos_events = pd.read_csv(opts.ts_pos_filepath)
    pos_events.loc[:,'EventTime'] = pos_events.ShockTime 
    neg_events = pd.read_csv(opts.ts_neg_filepath)
    
    if True:
        neg_events.loc[:,'LastMinute'] = neg_events.groupby(opts.unique_id_variable).tail(1)[opts.timestamp_variable]
        neg_events['LastMinute'] = neg_events.groupby(opts.unique_id_variable)['LastMinute'].bfill()
        
        # MSS = Multivariate State Sequence
        MSS_pos, MSS_neg = make_MSS(pos_events, neg_events, opts)
    else:
        MSS_pos, MSS_neg = load_MSS(opts)
        
    accuracy, precision, recall, f_measure, auc = pred_fold(MSS_pos, MSS_neg, opts)

    print("process time: {:.1f} min".format((time.time()-start)/60))
    
    return [accuracy, precision, recall, f_measure, auc, opts.classifier, opts.kernl, opts.folds]

