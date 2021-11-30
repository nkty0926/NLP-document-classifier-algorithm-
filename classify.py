'''
Title:           Document Classification
Files:           classify.py
Course:          CS540, Fall 2020

Author:          Tae Yong Namkoong
Email:           taeyong.namkoong@wisc.edu
References:      TA's Office Hours
'''

import math
import os
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
        :param vocab: to be processed
        :param filepath: file path for bow
        :return bag of words dictionary from a single document
    """
    bow = {} # declare dictionary to store
    with open(filepath, 'r', encoding='utf-8') as file: # open file
        for line in file: # process each line in file
            line = line.rstrip()
            if line in vocab: # check if already exist
                if line in bow:
                    bow[line] += 1 # increment occurence count if already in bow
                else:
                    bow[line] = 1 # else first occurence in bow
            else:
                if None in bow: # check for nonexistent
                    bow[None] += 1
                else:
                    bow[None] = 1
    return bow

def get_dir_path(directory):
    """
    This is a helper function that gets path for 2016 and 2020 sub directories
    :param directory: main directory path
    :return: a tuple that contains path for 2016 and 2020
    """
    subdirectory = os.listdir(directory) # get a list containing files in directory
    for label in subdirectory: # check for year label in subdirectory
        if label == "2016":
            dir_2016 = os.path.join(directory, label) # join path with label for 2016
        if label == "2020":
            dir_2020 = os.path.join(directory, label) # join path with label for 2020
    if dir_2016 is None or dir_2020 is None:
        print("subdirectory does not exist in this directory")
    return dir_2016, dir_2020 # return tuple to process later

def load_training_data(vocab, directory):
    """
    Create and return training set (bag of words Python dictionary + label) from the files in a training directory
    :param vocab: sorted vocabulary list from given files
    :param directory: directory to load the contents from
    :return: list of dictionaries
    """
    dataset = []
    dir_2016, dir_2020 = get_dir_path(directory) # load files from each directory
    file_2016 = os.listdir(dir_2016) # get list of all directories for 2016
    file_2020 = os.listdir(dir_2020) # get list of all directories for 2020

    path_2016 = [os.path.join(dir_2016, file) for file in file_2016] # join path for 2016 with file
    path_2020 = [os.path.join(dir_2020, file) for file in file_2020] # join path for 2020 with file

    for path in path_2020: # create dictionary for key, value for 2020
        temp = dict() # create dictionary for key, value for 2020
        temp['label'] = '2020'
        temp['bow'] = create_bow(vocab, path)
        dataset.append(temp) # append to bag of words dictionary


    for path in path_2016: # process path in 2016 label
        temp = dict()  # create dictionary for key, value for 2016
        temp['label'] = '2016'
        temp['bow'] = create_bow(vocab, path)
        dataset.append(temp) # append to bag of words dictionary

    return dataset

def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
        :param directory: the directory to process
        :param cutoff: exclude any word  which appear at a frequency strictly less than the cutoff
        :return sorted list of these word types.
    """
    vocab = [] # create dictionary to store label, and # of occurrences
    dic = {}
    label = get_dir_path(directory)
    for year in label: # iterate year label
        files = os.listdir(year) # get list of all directories given a year
        paths = [os.path.join(year, file) for file in files]
        for i in paths: # iterate each path
            with open(i, 'r', encoding='utf-8', errors='ignore') as file:
                for line in file: # iterate each word in each file
                    if line.strip() not in dic: # check if already existent in dic or not to increment count
                        dic[line.strip()] = 1
                    else:
                        dic[line.strip()] += 1
    for j in dic:
        if dic.get(j) >= cutoff: # check for cutoff threshold and filter out relevant results
            vocab.append(j) # then append to final sorted vocab list
    return sorted(vocab)

def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    :param training_data: loaded data set
    :param label_list: label years: 2016 and 2020
    :return: prior for years 2016 and 2020
    """
    smooth = 1 # smoothing factor
    logprob = {}
    length = len(training_data) # get length of training data
    for label in label_list: # iterate label in label list
        count = 0
        for i in range(length): # iterate and check for matching label to increment frequency of words
            if training_data[i].get('label') == label:
                count += 1
        logprob[label] = math.log((count + smooth) / (length + 2)) # use use add-1 smoothing method to calculate
    return logprob

def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing
    :param vocab: vocab
    :param training_data: training data
    :param label: years 2016 or 2020
    :return: dictionary consisting of the log conditional probability of all word types in a vocabulary
    """
    word_prob = {}
    for word in vocab: # iterate each word in vocab
        word_prob[word] = 1  # initialize all the values with 1 instead of add 1 later

    word_prob[None] = 1
    bows_to_add = [bow['bow'] for bow in training_data if bow['label'] == label]  # list with dict

    for bow in bows_to_add: # iterate bag of words
        for word in bow: # then for each iterate each word
            while bow[word] != 0:
                bow[word] -= 1
                word_prob[word] += 1

    total_count = 0
    for word in word_prob:
        total_count = total_count + word_prob[word]
    for word in word_prob: # use conditional probability of each word, given a label using add-1 smoothing
        word_prob[word] = math.log(word_prob[word] / total_count)
    return word_prob


##################################################################################
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    :param: training_directory: the training_directory passed in
    :param: cutoff: the threshold value for training set vocab
    :return: trained model in dict
    """
    vocab = create_vocabulary(training_directory, cutoff) # create vocab by passing training_directory and cutoff
    data = load_training_data(vocab, training_directory) # load data

    retval = {'vocabulary': vocab, 'log prior': prior(data, ['2020', '2016']),
          'log p(w|y=2020)': p_word_given_label(vocab, data, '2020'),
          'log p(w|y=2016)': p_word_given_label(vocab, data, '2016')} # return formatted dictionary

    return retval

def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    vocab = model['vocabulary']
    file_bow = create_bow(vocab, filepath) # create bow
    # calculate log probability for label
    p_2016 = calc(file_bow, model['log p(w|y=2016)'], model['log prior']['2016'])
    p_2020 = calc(file_bow, model['log p(w|y=2020)'], model['log prior']['2020'])
    retval = {}

    retval['log p(y=2020|x)'] = p_2020 # for 2020
    retval['log p(y=2016|x)'] = p_2016 # for 2016
    # output formatting based on which year greater
    if p_2016 > p_2020:
        retval['predicted y'] = '2016'
    else:
        retval['predicted y'] = '2020'

    return retval


'''
This function calculates log probabilities given some label 
    :param bow: dictionary with key, value pair that are words, count
    :param log_label: val given 
    :param log_prev: prior log val 
    :return log probability
'''
def calc(bow, log_label, log_prev):
    p = 0
    copy = {i: bow[i] for i in bow}
    for word in copy:
        while copy[word] != 0:
            copy[word] -= 1
            p += log_label[word]
    p += log_prev
    return p


