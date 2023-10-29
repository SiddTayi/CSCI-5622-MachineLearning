#!pip install graphviz 
#!pip install pydotplus

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import graphviz 
import pydotplus


from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


df_videos = pd.read_csv('data/Cleaned/Cleaned_videoData.csv')
df_videos.dropna(inplace = True, axis = 0)

df_videos_sampled = df_videos.sample(frac = 0.5) #, random_state=1)
cols = ['categoryId', 'view_count', 'likes', 'dislikes', 'comment_count', 'start_year']
df_videos[cols].head()


# Creating the target column: Popularity_category
df_videos_sampled['popularity_category'] = pd.cut(df_videos['view_count'], bins=[0, 100000, 1000000, 10000000, float('inf')], labels=['Low', 'Medium', 'High', 'Very'])

# Count of target classes
sns.catplot(data = df_videos_sampled, x = 'popularity_category', kind = 'count')


# Selecting the features for modeling
cols = ['categoryId', 'view_count', 'likes', 'dislikes', 'comment_count', 'start_year', 'start_month', 'popularity_category']
df = df_videos_sampled[cols]

df['popularity_category'].dropna(inplace = True, axis = 0)


# Train-Test Split
# One-hot encoding for 'categoryId'
data = df.copy()
data = pd.get_dummies(df, columns=['categoryId'], prefix='category')

# Mapping target object values to numeric 
class_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Very': 3
}

# Apply the mapping to create the 'popularity_category_encoded' column
data['popularity_category_encoded'] = data['popularity_category'].map(class_mapping)
data.dropna(inplace =True, axis = 0)
X = data[['category_Autos & Vehicles', 'category_Comedy',
       'category_Education', 'category_Entertainment',
       'category_Film & Animation', 'category_Gaming',
       'category_Howto & Style', 'category_Music', 'category_News & Politics',
       'category_Nonprofits & Activism', 'category_People & Blogs',
       'category_Pets & Animals', 'category_Science & Technology',
       'category_Sports', 'category_Travel & Events']]
Y = data[['popularity_category_encoded']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Modeling

# Manually assigning the class values
classes = [0, 1, 2, 3]

# Initialize the DecisionTreeClassifier with the specified classes
clf = DecisionTreeClassifier(random_state=42, class_weight=None, max_depth=None,
                            max_features=None, max_leaf_nodes=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, splitter='best',
                            ccp_alpha=0.0)


clf.fit(x_train, y_train)
# Use the model to predict the testset
pred = clf.predict(x_test)

# Classification report
report = metrics.classification_report(y_test, pred)
print(report)

cm = confusion_matrix(y_test, pred, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
plt.show()

print("Accuracy:", accuracy)

# Visualizing the tree
MyDT_Classifier = tree.DecisionTreeClassifier()
MyDT_Classifier = MyDT_Classifier.fit(x_test, y_test)
# Class names as integers
class_names = [0, 1, 2, 3]
TREE_Vis = tree.export_graphviz(MyDT_Classifier, 
                    feature_names=x_train.columns,  
                    class_names=[str(class_name) for class_name in class_names],
                    filled=True, rounded=True)  

graph = graphviz.Source(TREE_Vis)  
#graph 
#Export to pdf
DT_graph = pydotplus.graph_from_dot_data(TREE_Vis)
DT_graph.write_pdf("DT_videos.pdf")
DT_graph.write_png("DT_videos.png")

# -------------------------------------------------------------------------------------------------------------------------------------------------#

# SPLITTING CATEGORYID CLASSES TO COLUMNS 
df_copy = df.copy()
df_copy.dropna(inplace = True, axis = 0)

X = df_copy[['categoryId', 'view_count', 'likes', 'dislikes', 'comment_count',
       'start_year', 'start_month']]
Y = df_copy[['popularity_category']]

# Perform label encoding on the 'categoryId' column
label_encoder = LabelEncoder()
df_copy['categoryId'] = label_encoder.fit_transform(df_copy['categoryId'])
# df_copy['popularity_category_encoded'] = label_encoder.fit_transform(df_copy['popularity_category'])

class_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Very': 3
}

# Apply the mapping to create the 'popularity_category_encoded' column
df_copy['popularity_category_encoded'] = df_copy['popularity_category'].map(class_mapping)


# Define the features and the target variable
X = df_copy.drop(columns=['popularity_category', 'view_count'])
y = df_copy['popularity_category']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42, class_weight=None, max_depth=None,
                            max_features=None, max_leaf_nodes=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, splitter='best',
                            ccp_alpha=0.0)
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification_rep)

cm = confusion_matrix(y_test, y_pred, labels = dt_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = dt_classifier.classes_)
disp.plot()
plt.show()


# Tree viz
MyDT_Classifier = tree.DecisionTreeClassifier()
MyDT_Classifier = MyDT_Classifier.fit(X_test, y_test)

# Class names as integers
class_names = ['High', 'Low', 'Medium', 'Very']
TREE_Vis = tree.export_graphviz(MyDT_Classifier, 
                    feature_names=X_train.columns,  
                    class_names=[str(class_name) for class_name in class_names],
                    filled=True, rounded=True)  


DT_graph = pydotplus.graph_from_dot_data(TREE_Vis)
DT_graph.write_pdf("DT_videos_split.pdf")
DT_graph.write_png("DT_videos_split.png")

# -------------------------------------------------------------------------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------------------------------------------------------------------------#


# Splitting the data to 2 classes instead of 4
sampled_data = df_videos_sampled.copy()
sampled_data['popularity_category'] = pd.cut(df_videos['view_count'], bins=[0, 1000000, float('inf')], labels=['Low', 'High'])
sns.catplot(data = sampled_data, x = 'popularity_category', kind = 'count')

print(f'value counts: {sampled_data["popularity_category"].value_counts()}')

cols = ['categoryId', 'likes', 'dislikes', 'comment_count', 'start_year', 'start_month', 'popularity_category', ]
df = sampled_data[cols]

df['popularity_category'].dropna(inplace = True, axis = 0)
# One-hot encoding for 'categoryId'
data = df.copy()
df = pd.get_dummies(df, columns=['categoryId'], prefix='category')


# Label encoding for 'popularity_category'
# label_encoder = LabelEncoder()
# df['popularity_category_encoded'] = label_encoder.fit_transform(df['popularity_category'])
class_mapping = {
    'Low': 0,
    'High': 2
}

# Apply the mapping to create the 'popularity_category_encoded' column
df['popularity_category_encoded'] = df['popularity_category'].map(class_mapping)

df = df.drop_duplicates()
df = df.dropna(axis=0)

X = df[['category_Autos & Vehicles', 'category_Comedy',
       'category_Education', 'category_Entertainment',
       'category_Film & Animation', 'category_Gaming',
       'category_Howto & Style', 'category_Music', 'category_News & Politics',
       'category_Nonprofits & Activism', 'category_People & Blogs',
       'category_Pets & Animals', 'category_Science & Technology',
       'category_Sports', 'category_Travel & Events']]
Y = df[['popularity_category_encoded']]

# Converting Bool to int
X = X.astype(int)


# TRAIN-TEST
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)

##Use the model to predict the testset
pred = clf.predict(x_test)

report = metrics.classification_report(y_test, pred)

cm = confusion_matrix(y_test, pred, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
plt.show()

# -------------------------------------------------------------------------------------------------------------------------------------------------#

# Tree viz
MyDT_Classifier = tree.DecisionTreeClassifier()
MyDT_Classifier = MyDT_Classifier.fit(x_test, y_test)
# Class names as integers
class_names = ['High', 'Low', 'Other']
TREE_Vis = tree.export_graphviz(MyDT_Classifier, 
                    feature_names=x_train.columns,  
                    class_names=[str(class_name) for class_name in class_names],
                    filled=True, rounded=True)  

DT_graph = pydotplus.graph_from_dot_data(TREE_Vis)
DT_graph.write_pdf("DT_videos_2class.pdf")
DT_graph.write_png("DT_videos_2class.png")
