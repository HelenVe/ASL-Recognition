import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import editdistance
from utils import beam_decode, compute_acc
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
import os
import argparse
import tensorflow as tf
from jiwer import wer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

idx2letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k',
              11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v',
              22: 'w', 23: 'x', 24: 'y'
    , 25: 'z', 26: 'NULL'}
letter2idx = {v: k for k, v in idx2letter.items()}


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
            if len(label) == 13:
                label = label.pop()
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

    x = np.asarray(vec, dtype=object)
    x = pad_sequences(x, value=100.0, dtype='float32')
    y = np.asarray(list_of_y, dtype=object).astype('int32')

    return x, y


def greedy_decoding(pred):
    print("Greedy decoding pred shape", pred.shape)
    input_length = np.ones(pred.shape[0]) * pred.shape[1]
    pred = tf.squeeze(pred)
    ctc_decode = K.get_value(K.ctc_decode(pred, input_length=input_length, greedy=True))[0][0][:, :MAX_SEQ_LEN]

    decoded = ctc_decode
    # print("Greedy decoded", ctc_decode)
    output = []
    for i in range(len(decoded)):
        letter = idx2letter[int(decoded[i])]
        output.append(letter)
    return output


def beam_decoding(pred):
    input_len = [pred.shape[0]]

    pred = tf.squeeze(pred)

    # returns (decoded, log_probabilities)
    decoded = beam_decode(pred, beam_size=5, int_to_char=idx2letter, char_to_int=letter2idx)
    return decoded


def test(user):
    X_test, y_test = get_data("csv/fsvid.csv", "resnet/data/" + user + "/test/")
    print("X_test shape", X_test.shape)
    print("Y_test shape", y_test.shape)
    model = load_model("models/merged/rnn_model_bi_lstm")
    model = Model(model.get_layer("encoder_input").input, model.get_layer("dense").output)
    model.summary()

    # loss, acc = model.evaluate(X_test, y_test, verbose=2)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    MAX_SEQ_LEN = 13

    y_pred = model.predict(X_test)
    print(y_pred.shape)
    total_error = 0
    for i in range(len(y_pred)):
        pred = y_pred[i]
        word_label = []

        for letter in y_test[i]:
            word = idx2letter[int(letter)]
            word_label.append(word)

        print(pred.shape)
        output_txt_beam = beam_decoding(pred)
        print("Output txt Beam:", output_txt_beam)

        # Greedy Decoding
        # output_txt_greedy = greedy_decoding(pred)
        # print("Output text Greedy:", output_txt_greedy)

        print(word_label)

        error = wer(word_label, output_txt_beam)
        total_error += error
    #     accuracy = compute_acc(output_txt_beam, word_label)
    #     total_acc += accuracy
    #
    #     print("Word Accuracy", accuracy)

    print("WER", total_error / len(y_pred))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-u', '--users', default="andy", choices=['andy', 'drucie', 'rita', 'robin', 'all'],
                    help='choose user or train total')
    args = ap.parse_args()
    user = args.users
    test(user)
