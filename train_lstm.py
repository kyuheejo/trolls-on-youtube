import pandas as pd
import urllib.request
%matplotlib inline
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import argsparse
import os
import matplotlib.pyplot as plt


def preprocess(s):
  comment_result = []
  for comment in s:
      tokens = re.sub(emoji_pattern,"",comment[0])
      tokens = re.sub(han,"",tokens)
      comment_result.append([tokens])
  return comment_result 

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('Percentage of samples of length less than %s : %s'%(max_len, (cnt / len(nested_list))*100))


def main(args):
    os.mkdirs(args.save_path)
    df = pd.read_csv(args.data_path)
    df = df.iloc[:, 1:3]
    df = df.dropna()

    X = []
    for x in df['Comment']:
        X.append([x])

    y = []
    for i in df['Label']:
        y.append(i)

    train_X, test_X, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # remove emojis
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)

    # remove stopwords 
    han = re.compile(r'[ㄱ-ㅎㅏ-ㅣ!?~,".\n\r#\ufeff\u200d]')



    train_X = preprocess(train_X)
    test_X = preprocess(test_X)

    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    okt = Okt()
    X_train = []
    for sentence in train_X:
        temp_X = []
        temp_X = okt.morphs(sentence[0], stem=True) # tokenize
        temp_X = [word for word in temp_X if not word in stopwords] # remove stopwords 
        print(temp_X)
        X_train.append(temp_X)

    X_test = []
    for sentence in test_X:
        temp_X = []
        temp_X = okt.morphs(sentence[0], stem=True) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_test.append(temp_X)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # ============== Encoding =============== 

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    threshold = 3
    total_cnt = len(tokenizer.word_index) # number of words 
    rare_cnt = 0 # cound number of words with frequency less than threshold 
    total_freq = 0 # sum of all word frequency of train data
    rare_freq = 0 # sum of all word frequency of wors with frequency less than the threshold  

    # accept pair of word and frequency as key and value 
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # if word frequency is less than threshold 
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print('Size of vocabulary :',total_cnt)
    print('Words with frequency less than threshold %s: %s'%(threshold - 1, rare_cnt))
    print("Percentage of rare words:", (rare_cnt / total_cnt)*100)
    print("Percentage of frequency of rare words:", (rare_freq / total_freq)*100)

    # Remove words with freqeuncy less than 2 
    vocab_size = total_cnt - rare_cnt + 2
    print('Size of vocabulary:',vocab_size)

    tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Remove empty sample
    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)
    print(len(X_train))
    print(len(y_train))

    # Padding
    print('Length of longest comment :',max(len(l) for l in X_train))
    print('Length of average comment :',sum(map(len, X_train))/len(X_train))
    plt.hist([len(s) for s in X_train], bins=50)
    plt.xlabel('length of samples')
    plt.ylabel('number of samples')
    plt.show()

    max_len = 30
    below_threshold_len(max_len, X_train)
    X_train = pad_sequences(X_train, maxlen = max_len)
    X_test = pad_sequences(X_test, maxlen = max_len)

    # =============== Train model ================

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 100))
    model.add(LSTM(128))
    model.add(Dense(3, activation='softmax'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint(os.path.join(args.save_path, 'best_model_raw_v2.h5'), monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

    # ============== Save results ==================
    plt.plot(history.history['recall'], label = 'recall')
    plt.plot(history.history['acc'], label = 'accuracy')
    plt.plot(history.history['precision'], label = 'precision')
    plt.legend(['recall','accuracy','precision'])
    plt.title('LSTM metrics')
    plt.savefig(os.path.join(args.save_path, 'LSTM metrices'))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'validation loss'])
    plt.title('LSTM loss')
    plt.savefig(os.path.join(args.save_path, 'loss'))

    plt.plot(history.history['val_recall'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['val_precision'])
    plt.legend(['recall','accuracy','precision'])
    plt.title('LSTM validation metrics')
    plt.savefig(os.path.join(args.save_path, 'LSTM validation metrics'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('train LSTM')
    parser.add_argument('--data_path', default='./Comments_data.csv', type=str, help='path to comment data')
    parser.add_argument('--save_path', default='./result', type=str, help='file path to save results')
    args = parser.parse_args()
    main(args)