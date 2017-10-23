 # Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import codecs

import numpy as np
import re
import itertools
from collections import Counter
import os

# from gensim.models import word2vec



def load_data_and_labels(pos,neg):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    # Load data from files
    positive_examples = list(codecs.open(pos, "r", "utf-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    positive_examples = [s.split(" ") for s in positive_examples]
    negative_examples = list(codecs.open(neg, "r", "utf-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    negative_examples = [s.split(" ") for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0] for _ in positive_examples]
    negative_labels = [[1] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return x_text, y





def build_vocab(sentences):
    #TODO:this function got a bug !
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def build_input_data_with_word2vec(sentences, labels, word2vec):
    """Map sentences and labels to vectors based on a pretrained word2vec"""
    x_vec = []
    for sent in sentences:
        vec = []
        for word in sent:
            if word in word2vec:
                vec.append(word2vec[word])
            else:
                vec.append(word2vec['</s>'])
        x_vec.append(vec)
    x_vec = np.array(x_vec)
    y_vec = np.array(labels)
    return [x_vec, y_vec]

def pad_sentences(sentences):
    #TODO:bug to be fixed
    return 0;

def load_data_with_word2vec(word2vec):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    return build_input_data_with_word2vec(sentences_padded, labels, word2vec)


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_pretrained_word2vec(infile):
    if isinstance(infile, str):
        infile = open(infile)

    word2vec = {}
    for idx, line in enumerate(infile):
        if idx == 0:
            vocab_size, dim = line.strip().split()
        else:
            tks = line.strip().split()
            word2vec[tks[0]] = map(float, tks[1:])

    return word2vec

