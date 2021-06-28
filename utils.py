import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
import pandas as pd
import math
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K


# converts each word into a list of letters and returns an array of list of their classes
# for example the word "do" --> array([4, 15])

def return_classes(string):
    """
    takes a word, adds spaces between the letters and converts them to integers according to the dictionary made from
    index().
    then an array of ints is returned
    """

    text = list(string)  # list of letters
    # text = [" ".join(x) for x in text]   # add spaces between letters and EOS
    classes = [characters.index(char) for char in text]
    classes = np.array(classes)
    return classes


def index():
    """creates a dictionary from the words in the file"""
    # create index to word and word to index dictionary
    idx2letter = {}
    letter2idx = {}
    ix = 0
    for letter in characters:
        letter2idx[letter] = ix
        idx2letter[ix] = letter
        ix += 1
    return letter2idx, idx2letter


def plot(hist, num_epoch):
    """ Plot training accuracy and loss"""

    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    xc = range(num_epoch)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.style.use(['classic'])

    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    plt.style.use(['classic'])
    plt.show()


def iterative_levenshtein(s, t, costs=(1, 1, 1)):
    """
    Computes Levenshtein distance between the strings s and t.
    For all i and j, dist[i,j] will contain the Levenshtein
    distance between the first i characters of s and the
    first j characters of t
    s: source, t: target
    costs: a tuple or a list with three integers (d, i, s)
           where d defines the costs for a deletion
                 i defines the costs for an insertion and
                 s defines the costs for a substitution
    return:
    H, S, D, I: correct chars, number of substitutions, number of deletions, number of insertions
    """

    rows = len(s) + 1
    cols = len(t) + 1
    deletes, inserts, substitutes = costs

    dist = [[0 for x in range(cols)] for x in range(rows)]
    H, D, S, I = 0, 0, 0, 0
    for row in range(1, rows):
        dist[row][0] = row * deletes
    for col in range(1, cols):
        dist[0][col] = col * inserts

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row - 1][col - 1] + cost)
    row, col = rows - 1, cols - 1
    while row != 0 or col != 0:
        if row == 0:
            I += col
            col = 0
        elif col == 0:
            D += row
            row = 0
        elif dist[row][col] == dist[row - 1][col] + deletes:
            D += 1
            row = row - 1
        elif dist[row][col] == dist[row][col - 1] + inserts:
            I += 1
            col = col - 1
        elif dist[row][col] == dist[row - 1][col - 1] + substitutes:
            S += 1
            row, col = row - 1, col - 1
        else:
            H += 1
            row, col = row - 1, col - 1
    D, I = I, D
    return H, D, S, I


def compute_acc(preds, labels, costs=(7, 7, 10)):
    # cost according to HTK: http://www.ee.columbia.edu/~dpwe/LabROSA/doc/HTKBook21/node142.html

    if not len(preds) == len(labels):
        raise ValueError('# predictions not equal to # labels')
    Ns, Ds, Ss, Is = 0, 0, 0, 0
    for i, _ in enumerate(preds):
        H, D, S, I = iterative_levenshtein(preds[i], labels[i], costs)
        Ns += len(labels[i])
        Ds += D
        Ss += S
        Is += I
    try:
        acc = 100 * (Ns - Ds - Ss - Is) / Ns
    except ZeroDivisionError:
        raise ZeroDivisionError('Empty labels')
    return acc


def beam_decode(prob, beam_size, int_to_char, char_to_int, digit=False, blank_index=28):
    # prob: [seq_len, num_labels+1], numpy array
    seqlen = len(prob)
    beam_idx = np.argsort(prob[0, :])[-beam_size:].tolist()
    beam_prob = list(map(lambda x: math.log(prob[0, x]), beam_idx))
    beam_idx = list(map(lambda x: [x], beam_idx))

    for t in range(1, seqlen):
        topk_idx = np.argsort(prob[t, :])[-beam_size:].tolist()
        topk_prob = list(map(lambda x: prob[t, x], topk_idx))
        aug_beam_prob, aug_beam_idx = [], []

        for b in range(beam_size * beam_size):
            aug_beam_prob.append(beam_prob[b // beam_size])
            aug_beam_idx.append(list(beam_idx[b // beam_size]))

        # allocate
        for b in range(beam_size * beam_size):
            i, j = b / beam_size, b % beam_size
            aug_beam_idx[b].append(topk_idx[j])
            aug_beam_prob[b] = aug_beam_prob[b] + math.log(topk_prob[j])

        # merge
        merge_beam_idx, merge_beam_prob = [], []
        for b in range(beam_size * beam_size):
            if aug_beam_idx[b][-1] == aug_beam_idx[b][-2]:
                beam, beam_prob = aug_beam_idx[b][:-1], aug_beam_prob[b]
            elif aug_beam_idx[b][-2] == blank_index:
                beam, beam_prob = aug_beam_idx[b][:-2] + [aug_beam_idx[b][-1]], aug_beam_prob[b]
            else:
                beam, beam_prob = aug_beam_idx[b], aug_beam_prob[b]
            beam_str = list(map(lambda x: int_to_char[x], beam))
            if beam_str not in merge_beam_idx:
                merge_beam_idx.append(beam_str)
                merge_beam_prob.append(beam_prob)
            else:
                idx = merge_beam_idx.index(beam_str)
                merge_beam_prob[idx] = np.logaddexp(merge_beam_prob[idx], beam_prob)

        ntopk_idx = np.argsort(np.array(merge_beam_prob))[-beam_size:].tolist()
        beam_idx = list(map(lambda x: merge_beam_idx[x], ntopk_idx))
        for b in range(len(beam_idx)):
            beam_idx[b] = list(map(lambda x: char_to_int[x], beam_idx[b]))
        beam_prob = list(map(lambda x: merge_beam_prob[x], ntopk_idx))

    if blank_index in beam_idx[-1]:
        pred = beam_idx[-1][:-1]
    else:
        pred = beam_idx[-1]

    if digit is False:
        pred = list(map(lambda x: int_to_char[x], pred))

    return pred
