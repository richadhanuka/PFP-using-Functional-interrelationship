# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:11:37 2019

@author: shubh singh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:57:47 2019

@author: shubh singh
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import pandas as pd
from Metrics import metrics
import matplotlib.pyplot as plt
import pickle

df_train = pd.read_csv("Reduced_MLDA_400_train-bp_reducedLabel8.csv")

feature_size = 342#299   see above file bp-343
labels = df_train.shape[1] - feature_size#354 bp-562

'''
ls =[]
for i in range(df_train.shape[0]):
    if df_train.iloc[i,1:].sum()==0:
        ls.append(i)
        print(str(i) + "--" + str(df_train.iloc[i,1:].sum()))
df_train = df_train.drop(ls) # drops 26 rows with 0 annotations
'''

ds_train = df_train.values
X_train = df_train.iloc[:,:feature_size]
Y_train = df_train.iloc[:,feature_size:]


df_test = pd.read_csv("Reduced_MLDA_400_test-bp_reducedLabel8.csv")


'''
ls =[]
for i in range(df_test.shape[0]):
    if df_test.iloc[i,1:].sum()==0:
        ls.append(i)
        print(str(i) + "--" + str(df_test.iloc[i,1:].sum()))
df_test = df_test.drop(ls)
'''

ds_test = df_test.values
X_test = df_test.iloc[:,:feature_size]
Y_test = df_test.iloc[:,feature_size:]

filepath="model_best7.h5"


checkpoint = ModelCheckpoint(filepath,monitor="val_loss" ,verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# create model
model = Sequential()
model.add(Dense(600, input_dim=feature_size, activation='relu'))#
#model.add(Dense(400, activation='relu'))
#model.add(Dense(500, activation='relu'))
model.add(Dense(labels, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

 
# Fit the model

m = model.fit(X_train, Y_train , epochs=100, batch_size=32 ,validation_split=0.2, callbacks=callbacks_list, class_weight="balanced")

plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train Loss', 'Validation Loss'], loc='upper right')
plt.show() 

model = load_model(filepath)

Y_predicted = model.predict(X_test)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
# predict probabilities
lr_probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]


'''
thresholds =[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
plist, rlist, flist=[],[],[]
for t in thresholds:
    Y_predicted=model.predict(X_test)
    Y_predicted[Y_predicted >= t] = 1
    Y_predicted[Y_predicted < t] = 0
    avgPrecision, avgRecall, avgF1Score, F1Score = metrics(Y_predicted , Y_test.values)
    
    plist.append(avgPrecision)
    rlist.append(avgRecall)
    flist.append(F1Score)
    
import matplotlib.pyplot as plt
plt.plot(plist)
plt.plot(rlist)
plt.plot(flist)

plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], thresholds)
plt.grid()
plt.ylabel('percentage')
plt.xlabel('threshold')
plt.legend(['precision' , 'recall','f1-score'],  loc ='upper right')
plt.show()
'''

header=list(Y_test)  # header list
Y_predicted = pd.DataFrame(Y_predicted,columns=header)      #numpy array to dataframe with header as y_test
lr_probs = pd.DataFrame(Y_predicted,columns=header)


Y_predicted_expanded=Y_predicted.copy()
lr_probs_expanded=lr_probs.copy()                           #new Y_perdicted for expanded version, below work of expanding 
#Y_test_expanded=Y_test.copy()

#########################################################################################
from collections import defaultdict
pkl_file = open('dictionary8.pkl', 'rb')    #loading dictionary
dict_corr_col = pickle.load(pkl_file)
pkl_file.close()


for k,v in sorted(dict_corr_col.items(), reverse=True):
    #print(k)
    Y_predicted_expanded[str(k)] = Y_predicted_expanded[str(v[0])]
    lr_probs_expanded[str(k)]=lr_probs_expanded[str(v[0])]
 #   Y_test_expanded[str(k)]=Y_test_expanded[str(v)]                        #adding columns


ll=[]
for x in range(932):
    ll.append(str(x))
Y_predicted_expanded = Y_predicted_expanded[ll]
lr_probs_expanded = lr_probs_expanded[ll]

Y_predicted_expanded[Y_predicted_expanded >= 0.23] = 1
Y_predicted_expanded[Y_predicted_expanded < 0.23] = 0
#Y_test_expanded = Y_test_expanded[ll]
###############################################################################################
'''
Y_predicted_expanded1=Y_predicted.copy()
pkl_file = open('H:/major/newdata/0.8/dictionary.pkl', 'rb')    #loading dictionary
dict_corr_col1 = pickle.load(pkl_file)
pkl_file.close()
for k,v in dict_corr_col1:
    Y_predicted_expanded1[str(k)] = Y_predicted_expanded1[str(v)]
 #   Y_test_expanded[str(k)]=Y_test_expanded[str(v)]                        #adding columns

ll=[]
for x in range(589):
    if x !=408 and x!=409:                                #reordering header
       ll.append(str(x))
Y_predicted_expanded1 = Y_predicted_expanded1[ll]'''
##################################################################################################


d=pd.read_csv("test-bp.csv")

Y_test_expanded=d.iloc[:,1:]

from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(labels):
    precision[i], recall[i], _ = precision_recall_curve(Y_test_expanded.iloc[:, i],
                                                        lr_probs_expanded.iloc[:, i])
    average_precision[i] = average_precision_score(Y_test_expanded.iloc[:, i], lr_probs_expanded.iloc[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test_expanded.values.ravel(),
    lr_probs_expanded.values.ravel())
average_precision["micro"] = average_precision_score(Y_test_expanded, lr_probs_expanded,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

#Y_test_expanded = Y_test_expanded.drop(columns=['408','409'])
'''
print("EXPANDED WITH REVERSE DICTIONARY AND REMOVING COLUMN WITH TRANSITIVITY")
avgPrecision, avgRecall, avgF1Score, F1Score, hammingLoss = metrics(Y_predicted_expanded1.values, Y_test_expanded.values)

print("Average Precision : " + str(avgPrecision))

print("Average Recall : "  + str (avgRecall))

print("Average F1-Score : " + str(avgF1Score))

print("F1-score : " + str(F1Score))

print("Hamming Loss : " + str(hammingLoss))
'''
print("EXPANDED WITH NORMAL DICTIONARY WHERE COLUMN ADDED WITH THEIR ORIGINALITY FROMN TEST FILE")

avgPrecision, avgRecall, avgF1Score, F1Score = metrics(Y_predicted_expanded.values, Y_test_expanded.values)

#print("optimizer=Adam,epochs=300,batch_size=32")

print("Average Precision : " + str(avgPrecision))

print("Average Recall : "  + str (avgRecall))

print("Average F1-Score : " + str(avgF1Score))


print("F1-score : " + str(F1Score))



Y_predicted[Y_predicted >= 0.25] = 1
Y_predicted[Y_predicted < 0.25] = 0
print("LESS DATA ONE ")

avgPrecision, avgRecall, avgF1Score, F1Score = metrics(Y_predicted.values, Y_test.values)

#print("optimizer=Adam,epochs=300,batch_size=32")
print("Average Precision : " + str(avgPrecision))

print("Average Recall : "  + str (avgRecall))

print("Average F1-Score : " + str(avgF1Score))


print("F1-score : " + str(F1Score))
precision = dict()
recall = dict()
average_precision = dict()
for i in range(labels):
    precision[i], recall[i], _ = precision_recall_curve(Y_test.iloc[:, i],
                                                        lr_probs.iloc[:, i])
    average_precision[i] = average_precision_score(Y_test.iloc[:, i], lr_probs.iloc[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.values.ravel(),
    lr_probs.values.ravel())
average_precision["micro"] = average_precision_score(Y_test, lr_probs,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

