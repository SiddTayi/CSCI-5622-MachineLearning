import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load the data
data = pd.read_csv('data/Cleaned/Cleaned_videoData.csv')
data.dropna(inplace = True, axis = 0)

# Take a sample from the main dataset
sample_data = data.sample(frac = 0.2)

# Create a target variable with 4 classes
sample_data['popularity'] = pd.cut(sample_data['view_count'], bins=[0, 100000, 1000000, 10000000, float('inf')], labels=['Low', 'Medium', 'High', 'Very'])

# Count the number of classes
sns.catplot(data = sample_data, x = 'popularity', kind = 'count')


cols = ['categoryId', 'view_count', 'likes', 'dislikes', 'comment_count', 'start_year', 'popularity', ]
df = sample_data[cols]
df.dropna(inplace = True, axis = 0)

# Encoding
label_encoder = LabelEncoder()
df['categoryId'] = label_encoder.fit_transform(df['categoryId'])
category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Dependent and Independent variables
y = df[['popularity']]
X = df.drop('popularity', axis = 1)


# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Modeling
clf = GaussianNB()
clf.fit(x_train, y_train)

pred = clf.predict(x_test)
clf.score(x_train, y_train)


# Classification report
report = metrics.classification_report(y_test, pred)
print(report)


# Confusion matrix
cm = confusion_matrix(y_test, pred, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
plt.show()

print(cm)

# Accuracy
accuracy = accuracy_score(y_test, pred)
accuracy

# ---------------------------------------------------------------------------------------------------------------------------------#

## Splitting CategoryID class

cols = ['categoryId', 'likes', 'dislikes', 'comment_count', 'start_year', 'popularity']
df_split = df[cols].copy()

# Encoding
df_split_encoded = pd.get_dummies(df_split, columns=['categoryId'], prefix='category')
class_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Very': 3
}

# Apply the mapping to create the 'popularity_category_encoded' column
df_split_encoded['popularity_category_encoded'] = df_split_encoded['popularity'].map(class_mapping)
df_split_encoded.drop('popularity', inplace = True, axis = 1)

# Dependent - Independent cols
y = df_split_encoded[['popularity_category_encoded']]
x = df_split_encoded.drop('popularity_category_encoded', axis = 1)

# Train - Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train.shape, y_train.shape

# Modeling
clf = GaussianNB()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
clf.score(x_train, y_train)

# Classification report
report = metrics.classification_report(y_test, pred)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, pred, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, pred)
accuracy

# ---------------------------------------------------------------------------------------------------------------------------------#
## Considering only 2 classes
### Creating a target column with 2 classes instead of 4 to avoid class imbalance

sample_data_1 = sample_data.copy()
sample_data_1['popularity'] = pd.cut(sample_data['view_count'], bins=[0, 1000000, float('inf')], labels=['Low', 'High'])

sample_data_1.dropna(inplace = True, axis = 0)


# Count plot
sns.catplot(data = sample_data_1, x = 'popularity', kind = 'count')

# Subsetting the data for modeling
cols = ['categoryId', 'view_count', 'likes', 'dislikes', 'comment_count', 'start_year', 'popularity' ]
df = sample_data_1[cols]

# Modeling
label_encoder = LabelEncoder()
df['categoryId'] = label_encoder.fit_transform(df['categoryId'])
category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Independent and dependent vars
y = df[['popularity']]
X = df.drop('popularity', axis = 1)

# Train - Test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
x_train.shape, y_train.shape

# Modeling
clf = GaussianNB()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
clf.score(x_train, y_train)

# Clasification report
report = metrics.classification_report(y_test, pred)
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, pred, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
plt.show()

print(cm)

# Accuracy
accuracy = accuracy_score(y_test, pred)
accuracy
