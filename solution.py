#General imports
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from wordcloud import WordCloud

#Preprocessing
from sklearn.preprocessing import Normalizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw
from nltk.stem.snowball import ItalianStemmer
from sklearn.pipeline import Pipeline

#Classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


conversion = {0: 'neg', 1: 'pos'}

class StemTokenizer(object):
    def __init__(self):
        self.stemmer = ItalianStemmer()
        self.digits = re.compile("[0-9]")
        self.punct = re.compile('[%s]' % re.escape(string.punctuation))
        self.emoji = re.compile("["
                                u"\U0001F600-\U0001F64F"
                                u"\U0001F300-\U0001F5FF"
                                u"\U0001F680-\U0001F6FF"
                                u"\U0001F1E0-\U0001F1FF"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
    def __call__(self, document):
        stems = []
        for t in word_tokenize(document):
            t = t.strip()
            t = self.emoji.sub('', t)
            t = self.digits.sub('', t)
            t = self.punct.sub('', t)
            stem = self.stemmer.stem(t)
            if len(stem)!='' and len(stem)>2:
                stems.append(stem)
        return stems
        

def main():
    dev_data = load_file('dataset_winter_2020/development.csv', True)
    eval_data = load_file('dataset_winter_2020/evaluation.csv', False)

    stoplist = set(sw.words('italian'))
    print_wordcloud(dev_data, dev_data['class'], stoplist)

    stemTokenizer = StemTokenizer()

    vectorizer = TfidfVectorizer(tokenizer=stemTokenizer, lowercase=True, ngram_range=(1,3),
                                max_df=0.6, min_df=5, strip_accents='unicode')
    tfidf_X = vectorizer.fit_transform(dev_data['text'])
    data = vectorizer.transform(eval_data['text'])

    normalized_X = Normalizer().fit_transform(tfidf_X)
    normalized_data = Normalizer().transform(data)

    y = np.array(dev_data['class'])

    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, train_size=0.8, random_state=42, shuffle=True)
    y_pred, clf = SGD_Classifier(X_train, X_test, y_train, y_test)
    
    print('Running classifier...')
    clf.fit(normalized_X, y)
    y_pred = clf.predict(normalized_data)
    print('Writing results...')
    dump_to_file('predicted.csv', y_pred, list(eval_data.index))

    print_wordcloud(eval_data, y_pred, stoplist)

def load_file(filename, dev):
    df = pd.read_csv(filename, sep=',', na_values=[''])
    df = df.dropna()
    if dev:
        df['class'] = np.where(df['class']=='pos', 1, 0)
        pos = len(df[df['class']==1])
        neg = len(df[df['class']==0])
        values = ['neg', 'pos']
        y_pos = np.arange(2)
        counter = [neg, pos]
        colors = ['blue', 'red']
        plt.bar(y_pos, counter, align='center', alpha=0.5, color=colors)
        plt.xticks(y_pos, values)
        plt.ylabel('Count')
        plt.title('Review number per label')
        plt.show()
    return df

def dump_to_file(filename, labels, indexes):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Id', 'Predicted'])
        for i, label in enumerate(labels):
            writer.writerow([indexes[i], conversion[label]])

def print_wordcloud(data, labels, sw):
    file_positive = []
    file_negative = []
    for ind, value in enumerate(labels):
        if(value == 0):
            file_negative.append(data['text'][ind])
        if(value == 1):
            file_positive.append(data['text'][ind]) 

    print('Generating word cloud for negative reviews...')
    f0 = pd.Series(file_negative).str.cat(sep=' ')
    wordcloud = WordCloud().generate(f0)
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, stopwords=sw).generate(f0)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()  

    print('Generating word cloud for positive reviews...')
    f0 = pd.Series(file_positive).str.cat(sep=' ')
    wordcloud = WordCloud().generate(f0)
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, colormap='magma', stopwords=sw).generate(f0)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show() 

def XGB_Classifier(X_train, X_test, y_train, y_test):
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'F1 value with XGBClassifier: {f1}')
    return y_pred, clf

def NB_Classifier(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train.todense(), y_train)
    y_pred = clf.predict(X_test.toarray())
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'F1 value with Naiv Bayes: {f1}')
    return y_pred, clf

def DecisionTree_Classifier(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'F1 value with Decision Tree Classifier: {f1}')
    return y_pred, clf

def RandomForest_Classifier(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier()
    param_grid = {'n_estimators' : [200, 250, 300]}
    gridsearch = GridSearchCV(clf, param_grid, scoring='f1', cv=5)
    gridsearch.fit(X_train, y_train)
    print(f"Best parameters - n_estimators: {gridsearch.best_params_['n_estimators']}")
    model = gridsearch.best_estimator_
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'F1 value with Random Forest Classifier: {f1}')
    return y_pred, clf

def SVC_Classifier(X_train, X_test, y_train, y_test):
    #clf = SVC(C=2, kernel='rbf', gamma='scale') #Best parameter from grid search
    clf = SVC()
    param_grid = {'C' : [1, 2, 3],
                'kernel' : ['linear', 'rbf', 'sigmoid'],
                'gamma' : ['scale', 'auto']}
    gridsearch = GridSearchCV(clf, param_grid, scoring='f1', cv=5)
    gridsearch.fit(X_train, y_train)
    print(f"Best parameters - loss: {gridsearch.best_params_['C']}, penalty: {gridsearch.best_params_['kernel']}, gamma: {gridsearch.best_params_['gamma']}")
    clf = gridsearch.best_estimator_
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'F1 value with SVC: {f1}')
    return y_pred, clf

def SGD_Classifier(X_train, X_test, y_train, y_test):
    #clf = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.0001) #Best parameter from grid search
    clf = SGDClassifier(max_iter=1000)
    param_grid = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha' : [1e-4, 1e-3, 1e-2, 1e-1]}
    gridsearch = GridSearchCV(clf, param_grid, scoring='f1_weighted', cv=5)
    gridsearch.fit(X_train, y_train)
    print(f"Best parameters - loss: {gridsearch.best_params_['loss']}, penalty: {gridsearch.best_params_['penalty']}")
    print(f"Best parameters - alpha: {gridsearch.best_params_['alpha']}")
    clf = gridsearch.best_estimator_
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'F1 value with SGD: {f1}')
    return y_pred, clf

if __name__ == "__main__":
    main()