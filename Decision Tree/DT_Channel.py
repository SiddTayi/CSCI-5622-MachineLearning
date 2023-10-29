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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv('data/Cleaned/Cleaned_channelData.csv')
data.drop('Unnamed: 0', axis = 1, inplace = True)


# Set a threshold to bin the values to classes.
popularity_threshold = 5000000

# Create the target variable based on the threshold
data['popularity'] = (data['Subscribers'] > popularity_threshold).astype(int)

# Splitting Dependent and Independent variables
X = data[['Viewers', 'Videos_made', 'start_year', 'start_month', 'start_day', 'start_hour']]
# Define the target variable
y = data[['popularity']]


# Vis class count
sns.catplot(x = 'popularity', data = y, kind = 'count')

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
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

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels = dt_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = dt_classifier.classes_)
disp.plot()
plt.show()

# Tree viz
MyDT_Classifier = tree.DecisionTreeClassifier()
MyDT_Classifier = MyDT_Classifier.fit(X_test, y_test)

# Class names as integers
class_names = [0, 1]
TREE_Vis = tree.export_graphviz(MyDT_Classifier, 
                    feature_names=X_train.columns,  
                    class_names=[str(class_name) for class_name in class_names],
                    filled=True, rounded=True)  

graph = graphviz.Source(TREE_Vis)  
# graph

#Export to pdf and png
DT_graph = pydotplus.graph_from_dot_data(TREE_Vis)
DT_graph.write_pdf("Channel_Tree.pdf")
DT_graph.write_png("Channel_Tree.png")