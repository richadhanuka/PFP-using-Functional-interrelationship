from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import pandas as pd
from Metrics import metrics
import matplotlib.pyplot as plt
import pickle
from scipy import stats

filepath1 = "model_key.h5"
filepath2 ="model_value.h5"
model1 = load_model(filepath1)
model2 = load_model(filepath2)
model1.summary()
model2.summary()

df_train = pd.read_csv("train-bp-key.csv")
feature_size = 351
labels = df_train.shape[1]-feature_size

ds_train = df_train.values
X_train = ds_train[:,:feature_size]
Y_train = ds_train[:,feature_size:]


df_test = pd.read_csv("test-bp-key.csv")
ds_test = df_test.values
X_test = ds_test[:,:feature_size]
Y_test = ds_test[:,feature_size:]

from keras.models import Model
layer_name='dense_46'
im1 = Model(inputs=model1.input, outputs=model1.get_layer(layer_name).output)
X1_test=im1.predict(X_test)
X1_train = im1.predict(X_train)


df_train = pd.read_csv("train-bp-value.csv")

feature_size = 351#299   see above file 
labels = df_train.shape[1]-feature_size#354

ds_train = df_train.values
X_train = ds_train[:,:feature_size]
Y_train = ds_train[:,feature_size:]


df_test = pd.read_csv("test-bp-value.csv")

ds_test = df_test.values
X_test = ds_test[:,:feature_size]
Y_test = ds_test[:,feature_size:]


from keras.models import Model
layer_name='dense_48'
im2 = Model(inputs=model2.input, outputs=model2.get_layer(layer_name).output)
X2_test=im2.predict(X_test)
X2_train=im2.predict(X_train)

X_test=pd.concat([pd.DataFrame(X1_test), pd.DataFrame(X2_test)],axis=1)
X_train=pd.concat([pd.DataFrame(X1_train), pd.DataFrame(X2_train)],axis=1)

df_train = pd.read_csv("train-bp.csv")
Y_train =  df_train.iloc[:,1:]

df_test = pd.read_csv("test-bp.csv")
Y_test =  df_test.iloc[:,1:]


# k fold
tList,precList,recallList,f1List = [],[],[],[]
from sklearn.model_selection import KFold
fold=5
kfold = KFold(n_splits=fold, shuffle=True)
cvscores = []
i=0
X_train = X_train.values
Y_train = Y_train.values

for train, test in kfold.split(X_train, Y_train):
    filepath="model_windows" + str(i)+".h5"
    checkpoint = ModelCheckpoint(filepath,monitor="val_loss" ,verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model = Sequential()
    model.add(Dense(932, input_dim=906, activation='sigmoid'))
    #model.add(Dense(932, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    m = model.fit(X_train[train], Y_train[train] , verbose=1,epochs=500,validation_split=0.2, batch_size=64, callbacks=callbacks_list, class_weight="balanced")
    model = load_model(filepath)
    thresholds = [0.1,0.14, 0.16, 0.18, 0.20,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.4,0.5,0.6]
    plist,rlist,flist = [],[],[]
    for t in thresholds:
        Y_predicted = model.predict(X_train[test])
        Y_predicted[Y_predicted>= t] = 1
        Y_predicted[Y_predicted< t] = 0
        avgPrecision, avgRecall, avgF1Score, F1Score = metrics(Y_predicted, Y_train[test])
        plist.append(avgPrecision)
        rlist.append(avgRecall)
        flist.append(F1Score)
    
    li=[]
    for ix in range(len(thresholds)):
        if plist[ix]>=rlist[ix]:
            li.append(ix)
            
    znew = np.argmax([flist[l] for l in li])
    maxIndex=li[znew]

    #maxIndex = np.argmax(flist)
    print("maxIndex:" + str(maxIndex))
    t = thresholds[maxIndex]
    tList.append(t)
    print("threshold:" + str(t))
    Y_predicted = model.predict(X_train[test])
    Y_predicted[Y_predicted>= t] = 1
    Y_predicted[Y_predicted< t] = 0
    avgPrecision, avgRecall, avgF1Score, F1Score = metrics(Y_predicted, Y_train[test])
    precList.append(avgPrecision)
    recallList.append(avgRecall)
    f1List.append(F1Score)
    
    plt.close()
    plt.plot(plist)
    plt.plot(rlist)
    plt.plot(flist)
    xtick=list(range(15))
    plt.xticks(xtick, thresholds)
    plt.grid()
    plt.ylabel('percentage')
    plt.xlabel('threshold')
    plt.legend(['precision','recall','f1-score'], loc = 'upper right')
    #plt.show()
    plt.savefig("PRgraph_"+str(i)+".png")
    i=i+1
    
precList_test,recallList_test,f1List_test, tList_test =[],[],[],[]
thres_mean = np.mean(tList)
thres_median = np.median(tList)
thres_mode = stats.mode(tList)
thres =[thres_mean,thres_median,list(thres_mode[0])]
# adam, dense 100, epoch 100, batch 128

for i in range(fold):
    model = Sequential()
    model.add(Dense(932, input_dim=906, activation='sigmoid'))
    #model.add(Dense(932, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    filepath="modelwindowstest_"+str(i)+".h5"
    checkpoint = ModelCheckpoint(filepath,monitor="val_loss" ,verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    m = model.fit(X_train, Y_train , epochs=500, batch_size=64  ,validation_split=0.2, callbacks=callbacks_list, class_weight="balanced")
    
    plt.plot(m.history['loss'])
    plt.plot(m.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train Loss', 'Validation Loss'], loc='upper right')
    plt.show() 
    
    model = load_model(filepath)
    for k in range(len(thres)):
        
        Y_predicted = model.predict(X_test)
    
        threshold = thres[k]
        Y_predicted[Y_predicted >= threshold] = 1
        Y_predicted[Y_predicted < threshold] = 0
        
        avgPrecision, avgRecall, avgF1Score, F1Score = metrics(Y_predicted, Y_test.values)
        tList_test.append(threshold)
        precList_test.append(avgPrecision)
        recallList_test.append(avgRecall)
        f1List_test.append(F1Score)

#aupr
filepath='modelwindowstest_4.h5'
model = load_model(filepath)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
# predict probabilities
lr_probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
#lr_probs = lr_probs[:, 1]
        
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(labels):
    precision[i], recall[i], _ = precision_recall_curve(Y_test.iloc[:, i],
                                                        lr_probs[:, i])
    average_precision[i] = average_precision_score(Y_test.iloc[:, i], lr_probs[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.values.ravel(),
    lr_probs.ravel())
average_precision["micro"] = average_precision_score(Y_test, lr_probs,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))