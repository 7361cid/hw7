import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model

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

X_train_sample = X_train[:10]
y_train_sample = y_train[:10]
clf = LogisticRegression()
clf.w = np.random.randn(X_train_sample.shape[1]+1) * 2

#loss, grad = clf.loss(X_train_sample, y_train_sample, 0.0)
#f = lambda w: clf.loss(X_train_sample, y_train_sample, 0.0)[0]
#grad_numerical = grad_check_sparse(f, clf.w, grad, 10)

clf.train(X_train, y_train, verbose=True)
print("Train finish")
print("Train f1-score = %.3f" % accuracy_score(y_train, clf.predict(X_train)))
print("Test f1-score = %.3f" % accuracy_score(y_test, clf.predict(X_test)))

#clf = linear_model.SGDClassifier(max_iter=2000, random_state=42, loss="log_loss", penalty="l2", alpha=1e-3, eta0=1.0, learning_rate="constant")
#clf.fit(X_train, y_train)
print("Train accuracy = %.3f" % accuracy_score(y_train, clf.predict(X_train)))
print("Test accuracy = %.3f" % accuracy_score(y_test, clf.predict(X_test)))