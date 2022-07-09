import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from dmia.gradient_check import *
from dmia.classifiers.logistic_regression import LogisticRegression

train_df = pd.read_csv('./data/train.csv')
review_summaries = list(train_df['Reviews_Summary'].values)
review_summaries = [l.lower() for l in review_summaries]

vectorizer = TfidfVectorizer()
tfidfed = vectorizer.fit_transform(review_summaries)

X = tfidfed
y = train_df.Prediction.values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

X_train_sample = X_train[:10]   #
y_train_sample = y_train[:10]
clf = LogisticRegression()
# + 2 потому что append_biases увеличивает кол-во измерений на 1, а весов итак нужно на 1 больше из-за свободного члена
clf.w = np.random.randn(X_train_sample.shape[1]+2) * 2
print("Main STEР1")
# loss, grad = clf.loss(X_train_sample, y_train_sample, 0.0)
# print(f"Main STEР2 {loss} \n type(grad) {type(grad)} {grad}")
# f = lambda w: clf.loss(X_train_sample, y_train_sample, 0.0)[0]
# grad_numerical = grad_check_sparse(f, clf.w, grad, 10)
print(f"f clf.w {clf.w.shape}")
clf.train(X_train, y_train, ...)
print("Train f1-score = %.3f" % accuracy_score(y_train, clf.predict(X_train)))
print("Test f1-score = %.3f" % accuracy_score(y_test, clf.predict(X_test)))
