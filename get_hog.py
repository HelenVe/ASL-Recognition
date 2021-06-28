import numpy as np
import pandas as pd
import h5py
import cv2
import os
import csv
import scipy.io
import shutil
import itertools
import argparse
from PIL import Image

# code to get the HoG features foe every user and save them to "hog" folder

idx2letter = {-1: "", 0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
              12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w',
              23: 'x', 24: 'y', 25: 'z', 26: 'SIL_START', 27: 'SIL_END'}

def get_hog(user):
    f = scipy.io.loadmat('hog_info_'+user+'.mat')  # read file

    # get variables
    hog_features = np.array(f.get('X'))
    len_frames_fold = np.array(f.get('T'))  # length of frames in each fold
    len_word_fold = np.array(f.get('S'))  # length of words in each fold (so frames)
    word_sequence = np.array(f.get('W'))
    letter_labels = np.array(f.get('Lletter'))
    phono = np.array(f.get('Lph'))

    # print("Hog:", hog_features[0])
    # print("Word sequence ", word_sequence[0][0].shape)
    # print(letter_labels[0])
    # print(letter_labels[0][1].shape)
    # print(phono[1].shape)

    # list of word lengths
    len_frames_fold = len_frames_fold.squeeze()
    # print(len_word_fold[0])
    # word_lengths = []

    # for i in range(len(len_word_fold[0])):
    #     word_lengths.append(list(len_word_fold[0][i].squeeze()))
    # word_lengths = list(itertools.chain.from_iterable(word_lengths))

    path = "hog/"+user

    # code to get and seperate the hogs into their folds

    count = 0
    for index, val in enumerate(len_frames_fold):

        print(index, val)
        save_path = os.path.join(path, "fold_"+str(index)+"/")
        print(save_path)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for i in range(0, val):
            hog = hog_features[0][index][:, i]
            np.save(save_path + "hog_" + str(count).zfill(5) + ".npy", hog)
            count += 1

    xternal = 1  # counter for the number of words in each fold so 54

    for index, fold in enumerate(sorted(os.listdir(path))):
        word_per_fold = len_word_fold[0][index].squeeze()

        # create new folds
        save_fold = os.path.join(path, "new_fold_" + str(index))
        if not os.path.isdir(save_fold):
            os.mkdir(save_fold)

        # keep the path of the initial folds
        fold_path = os.path.join(path, fold)

        # for the array of lengths for the current fold
        for length in word_per_fold:

            count = 0  # counter for the total number of length iterations
            # for the files in the current fold
            for files in sorted(os.listdir(fold_path)):

                # inside each new fold seperate the total hogs into hog of words
                # like new_fold_0/hog_0..hog_54
                save_path = os.path.join(save_fold, "frames_" + str(xternal).zfill(4))

                if not os.path.isdir(save_path):
                    os.mkdir(save_path)

                file_path = os.path.join(fold_path, files)

                if count < length:
                    shutil.move(file_path, save_path)
                    count += 1
                else:
                    break
            xternal += 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-u', '--users', default="andy", choices=['andy', 'drucie', 'rita', 'robin'],
                    help='choose user')
    args = ap.parse_args()
    user = args.users
    get_hog(user)