# This is  a project to create a machine learning model for fast credit card fraud detection 
from __future__ import print_function
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from snapml import DecisionTreeClassifier as snamplDecisionTree
# data link
#url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
raw_data = pd.read_csv("creditcard.csv")
# Our data set has 284807 observations but in a real scenario a bank usually would have a larger data set compared to ours ?
# To simulate such an experience we will inflate our data set with obersvations making it size 10 times bigger than the original
multiplier = 10
inflated_data = pd.DataFrame(np.repeat(raw_data.values,multiplier,axis=0),columns=raw_data.columns)
# np.repeat will create a numpy array containing numpy values from raw_data dataframe repeat them (multiplier) number of times
# axis = 0 means row this gets executed and then is used as a parameter for the function pd.DataFrame()
# pd.DataFrame(n,columns=otherDataFrame.columns) takes arguement n converts it into pandas DataFrame the columns parameter ensures
# that the new data frame has the same column as the set columns from original ? or any existing dataframe
labels = inflated_data.Class.unique()
# takes unique values from the Class column of the data set and assigns it to label
sizes = inflated_data.Class.value_counts().values
# takes the count for unique values in the Class column of the data frame and stores it as a anumpy array
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
#plt.savefig("results.png")
# un-comment the above command and run it only once 
# plt.hist(values,number of bins(as in towers),histogram type,color of the bins) is the syntax for a histogram
plt.clf()
# clears the plot for a new plot
plt.hist(inflated_data.Amount.values,6,histtype='bar',facecolor='g')
#plt.savefig("Amounts.png")
print("Minimum amount value is ", np.min(inflated_data.Amount.values))
print("Maximum amount value is ", np.max(inflated_data.Amount.values))
print("90% of the transactions have an amount less or equal than ", np.percentile(raw_data.Amount.values, 90))
inflated_data.iloc[:,1:30] = StandardScaler().fit_transform(inflated_data.iloc[:,1:30])
# iloc selects rows and columns : means all rows 1:30 means from column 1 to 30 
data_matrix = inflated_data.values
x = data_matrix[:,1:30]
y = data_matrix[:,30]
x= normalize(x,norm='l1')
print(x.shape,y.shape)
# [:,1] is going to select all rows and then second column 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
print('X train shape ',x_train.shape,'Y train shape ',y_train.shape)
print('X test shape ',x_test.shape,'Y test shape ',y_test.shape)
train_weight = compute_sample_weight('balanced',y_train)
# compute_sample_weight function calculates weight for training sample balanced option tells the function to to automatically 
# calculate weights inversely proportional to class frequencies. This approach assigns higher weights to samples from 
# underrepresented classes, making the model more sensitive to minority classes.
decision_tree = DecisionTreeClassifier(max_depth=4,random_state=32)
# max_depth 4 ensures the tree stays shallow and and is not very long solving the problem of over fitting 
# if tree is short over fitting doesn't happen if it is long it handles complex data more profoundly 
t0 = time.time()
decision_tree.fit(x_train,y_train,sample_weight=train_weight)
decision_tree_time = time.time() - t0
print('Scikit Learn training time = {0:0.5f}'.format(decision_tree_time))
snap_Tree = snamplDecisionTree(max_depth=4,random_state=45,n_jobs=12)
t1 = time.time()
snap_Tree.fit(x_train,y_train,sample_weight=train_weight)
snapmlTime = time.time() - t1
print('SnapML training time = {0:0.5f}'.format(snapmlTime))


