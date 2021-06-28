import numpy as np
import pandas as pd
import h5py
import cv2
import os
import csv
import scipy.io
import argparse
from PIL import Image

# Code to extract the frames from the .mat file for every user and save them
# The name and offset has to be specified for every one.
# Some useful entries are saved ot a csv file


idx2letter = {-1: "", 0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
              12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w',
              23: 'x', 24: 'y', 25: 'z', 26: "", 27: ""}


def get_data(user, offset):
    f = h5py.File('image128_color0_'+user+'.mat', 'r')  # read file

    frames = np.array(f.get('X'))
    frame_lengths = np.array(f.get('T'))  # for each word, sums to the total number of frames
    word_sequence = np.array(f.get('W'))
    label = np.array(f.get('L'))
    D = np.array(f.get('D'))

    # print("Frames shape", frames.shape)
    # print("Frame lengths", frame_lengths.shape)
    # print("Word sequence shape", word_sequence.shape)
    # print("Label shape", label.shape)
    # print("Number of instances in each fold", D.shape)
    # print(D)

    offset = offset  # the index from the previous folder in order to save the frames in order and not start from 0 for
    # every user
    count = 0
    for index, length in enumerate(frame_lengths):
        labels = []
        name = "frames_" + str(index + 1 + offset).zfill(4)
        print(name)
        save_path = os.path.join("data/"+user, name)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for i in range(0, int(length)):
            count += 1
            img = Image.fromarray(frames[i])
            img = img.convert('RGB')

            img.save(save_path + "/" + str(count).zfill(5) + ".jpg", "JPEG")

            labels.append(int(label[i]))

        # # append to csv
        # with open("csv/fsvid.csv", "a") as csvfile:
        #
        #     csvwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        #     letter_label = [idx2letter[num] for num in word_sequence[index]]
        #     letters_with_sil = [idx2letter[la] for la in labels]
        #     csvwriter.writerow([index + offset, name, int(length), list(word_sequence[index]), list(letter_label),
        #                         list(letters_with_sil), list(labels)])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-u', '--users', default="andy", choices=['andy', 'drucie', 'rita', 'robin'],
                    help='choose user')
    args = ap.parse_args()
    user = args.users

    # we know from the info how many words belong to each user, and we want to save their frames in folders like:
    # frames_0001, so we have to specify the offset in order to save the correct name. Remember that in the get data
    # function the offset + 1 is added

    if user == "andy":
        offset = 0
    elif user == "drucie":
        offset = 547
    elif user == "rita":
        offset = 1114
    elif user == "robin":
        offset = 1680
    else:
        offset = -1
        print("Wrong user")
        exit()

    get_data(user, offset)
