import numpy as np
import pandas as pd
import os
import time
import ast
import random
import shutil
from utils import return_classes
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import time
import random
import shutil
from utils import return_classes
from tensorflow.keras.utils import normalize

random.seed(1)
# This is a Data generator that feeds the resnet features to the train function
#  if a folder has 15 images so 15 feature vectors of dim 512, the input to the fit function is reshape
#  to be (batch_Size, 15, 512). So the  general shape is (batch_size, number_of_vectors, 512)

idx2letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k',
              11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v',
              22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: 'NULL'}


def get_data(csv_name, path_to_vectors):
    df = pd.read_csv(csv_name)
    df = df.drop_duplicates()
    df['sequence'].fillna("missing")
    filename_list = sorted(df['filename'].tolist())

    list_of_y, vec = [], []

    for folder in sorted(os.listdir(path_to_vectors)):
        if folder in filename_list:
            idx = filename_list.index(folder)
            label = df.iloc[idx]['sequence']
            label = label.strip('][').split(', ')
            label = [int(i) for i in label]
            label = [26 if i == -1 else i for i in label]
        else:
            continue

        data = []
        folder_path = path_to_vectors + folder + "/"

        for file in sorted(os.listdir(folder_path)):
            file_path = folder_path + file
            array = np.load(file_path).squeeze()
            data.append(array)  # append to the data to it
        vec.append(np.array(data))  # list of feature vectors
        list_of_y.append(np.array(label, dtype=object))  # list of letters to int for each word, for example [1, 5, 3]

    x = np.array(vec, dtype=object)
    x = pad_sequences(x, value=100.0, dtype='float32', padding='post')

    # although the documentation states that label lengths are 12, i found some that have length 13, so pad the rest
    # with the value 26 and add a label masking layer to the model

    y = pad_sequences(list_of_y, padding='post', value=26, dtype='int32')
    y = np.asarray(y, dtype=object).astype('int32')

    input_length = []
    label_length = []
    # for i in range(len(x)):
    #     input_length.append(np.array(len(x[i])))
    #     label_length.append(np.array(len(y[i])))
    return x, y, input_length, label_length


class Generator(Sequence):
    def __init__(self, path, directory, batch_size, user, total_train):
        self.path = str(path)
        self.directory = "/" + directory
        self.batch_size = batch_size
        self.user = user
        self.total_train = total_train
        whole_path = os.path.join(self.path + self.directory)

    def __len__(self):
        whole_path = os.path.join(self.path + self.directory)
        return int(np.floor(len(os.listdir(whole_path)) // self.batch_size))

    def __getitem__(self, index):
        x, y, input_len, label_len = self.__load_dataset()

        batch_x = np.empty((self.batch_size, *x[index].shape))
        batch_y = np.empty((self.batch_size, *y[index].shape))

        'Generates a batch of data and gets reinitialized'
        batch_x = x[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = y[index * self.batch_size:(index + 1) * self.batch_size]

        print(batch_x.shape, batch_y.shape)
        print(batch_y)

        return [batch_x, batch_y]

    def __load_dataset(self):

        path_to_vectors = self.path + self.directory + "/"

        if "train" in path_to_vectors:
            if self.total_train:
                csv_name = "csv/fsvid_train_total.csv"
            else:
                csv_name = "csv/fsvid_train_" + self.user + ".csv"
            X_train, y_train, input_len, label_len = get_data(csv_name, path_to_vectors)
            return X_train, y_train, input_len, label_len

        elif "dev" in path_to_vectors:
            if self.total_train:
                csv_name = "csv/fsvid_dev_total.csv"
            else:
                csv_name = "csv/fsvid_dev_" + self.user + ".csv"

            X_dev, y_dev, input_len, label_len = get_data(csv_name, path_to_vectors)
            return X_dev, y_dev, input_len, label_len

        elif "test" in path_to_vectors:
            if self.total_train:
                csv_name = "csv/fsvid_test_total.csv"
            else:
                csv_name = "csv/fsvid_test_" + self.user + ".csv"

            X_test, y_test, input_len, label_len = get_data(csv_name, path_to_vectors)
            return X_test, y_test, input_len, label_len

        else:
            print("Generator: No partition train, dev or test found!")
