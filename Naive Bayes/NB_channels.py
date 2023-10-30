import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv('data/Cleaned/Cleaned_channelData.csv')
data.drop('Unnamed: 0', axis = 1, inplace = True)

data.drop('channel_description', inplace = True, axis = 1)

# Subsetting
cols = ['Subscribers','Viewers','Videos_made','start_year','start_month','start_day']
df = data[cols].copy()
df.head()

# Creating the target var
popularity_threshold = 5000000
# Create the target variable based on the threshold
df['popularity'] = (df['Subscribers'] > popularity_threshold).astype(int)

# Count plot
sns.catplot(data = df, x = 'popularity', kind = 'count')

# Dependent and Independent var
X = df[['Viewers', 'Videos_made', 'start_year', 'start_month', 'start_day']]
y = df[['popularity']]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Modeling
clf = GaussianNB()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
clf.score(X_train, y_train)

# Classification report
report = metrics.classification_report(y_test, pred)
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, pred, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, pred)
accuracy


# Multinomial NB
clf = MultinomialNB(alpha=1)
clf.fit(X_train, y_train)

pred = clf.predict(x_test)
clf.score(X_train, y_train)

report = metrics.classification_report(y_test, pred)
print(report)

cm = confusion_matrix(y_test, pred, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
plt.show()

accuracy = accuracy_score(y_test, pred)
accuracy



