import pandas as pd
import numpy as np
import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from dmia.classifiers.logistic_regression_v2 import LogisticRegression

train_df = pd.read_csv('./data/train.csv')
review_summaries = list(train_df['Reviews_Summary'].values)
review_summaries = [l.lower() for l in review_summaries]

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(review_summaries)
y = train_df.Prediction.values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
clf = LogisticRegression()
clf.train(X_train, y_train, verbose=True, learning_rate=1.0, num_iters=100, batch_size=256, reg=1e-3)
print("Train finish")
print("Train accuracy = %.3f" % accuracy_score(y_train, clf.predict(X_train)))
print("Test accuracy = %.3f" % accuracy_score(y_test, clf.predict(X_test)))
# Получите индексы фичей
pos_features = np.argsort(clf.w)[:5]
neg_features = np.argsort(clf.w)[-5:]
# Выведите слова
fnames = vectorizer.get_feature_names_out()
print(f"pos_features {pos_features}  {pos_features.shape}")
print(f"neg_features {neg_features}  {neg_features.shape}")
print([fnames[p-1] for p in pos_features])
print([fnames[n-1] for n in neg_features])
