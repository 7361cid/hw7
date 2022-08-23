import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import tqdm  # interactive progress bar
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from dmia.classifiers.logistic_regression import LogisticRegression

train_df = pd.read_csv('./data/train.csv')
review_summaries = list(train_df['Reviews_Summary'].values)
review_summaries = [l.lower() for l in review_summaries]

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(review_summaries)
y = train_df.Prediction.values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
clf = LogisticRegression()
clf.train(X_train, y_train, verbose=True, learning_rate=1.0, num_iters=100, batch_size=256, reg=1e-3)
print("Train accuracy = %.3f" % accuracy_score(y_train, clf.predict(X_train)))
print("Test accuracy = %.3f" % accuracy_score(y_test, clf.predict(X_test)))
# Получите индексы фичей
pos_features = np.argsort(clf.w)[:5]
neg_features = np.argsort(clf.w)[-5:]
# Выведите слова
fnames = vectorizer.get_feature_names_out()
print([fnames[p-1] for p in pos_features])
print([fnames[n-1] for n in neg_features])

# построение графиков
clf = LogisticRegression()
train_scores = []
test_scores = []
num_iters = 1000

for i in tqdm.trange(num_iters):
    # Сделайте один шаг градиентного спуска с помощью num_iters=1
    clf.train(X_train, y_train, learning_rate=1.0, num_iters=1, batch_size=256, reg=1e-3)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(10,8))
plt.plot(train_scores, 'r', test_scores, 'b')
plt.show()
