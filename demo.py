import glob
import os

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from nbsvm import NBSVM


def load_imdb():
    print("Vectorizing Training Text")

    train_pos = glob.glob(os.path.join('aclImdb', 'train', 'pos', '*.txt'))
    train_neg = glob.glob(os.path.join('aclImdb', 'train', 'neg', '*.txt'))

    vectorizer = CountVectorizer('filename', ngram_range=(1, 3), binary=True)
    X_train = vectorizer.fit_transform(train_pos+train_neg)
    y_train = np.array([1]*len(train_pos)+[0]*len(train_neg))

    print("Vectorizing Testing Text")

    test_pos = glob.glob(os.path.join('aclImdb', 'test', 'pos', '*.txt'))
    test_neg = glob.glob(os.path.join('aclImdb', 'test', 'neg', '*.txt'))

    X_test = vectorizer.transform(test_pos + test_neg)
    y_test = np.array([1]*len(test_pos)+[0]*len(test_neg))

    return X_train, y_train, X_test, y_test

def main():

    X_train, y_train, X_test, y_test = load_imdb()

    print("Fitting Model")

    mnbsvm = NBSVM()
    mnbsvm.fit(X_train, y_train)
    print('Test Accuracy: %s' % mnbsvm.score(X_test, y_test))

if __name__ == '__main__':
    main()
