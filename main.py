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
# data link
#url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
raw_data = pd.read_csv("creditcard.csv")
print("There are " + str(len(raw_data)) + " observations in the credit card fraud dataset.")
print("There are " + str(len(raw_data.columns)) + " variables in the dataset.")
# Our data set has 284807 observations but in a real scenario a bank usually would have a larger data set compared to ours ?
# To simulate such an experience we will inflate our data set with obersvations making it size 10 times bigger than the original
multiplier = 10
inflated_data = pd.DataFrame(np.repeat(raw_data.values,multiplier,axis=0),columns=raw_data.columns)
