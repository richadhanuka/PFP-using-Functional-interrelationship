from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import pandas as pd
from Metrics import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statistics
from scipy import stats

df_train = pd.read_csv("Reduced_MLDA_400_train-bp.csv",header=None)
train, validate = train_test_split(df_train, test_size=0.2)
feature_size = 351
labels = 932
'''
ls =[]
for i in range(df_train.shape[0]):
    if df_train.iloc[i,1:].sum()==0:
        ls.append(i)
        print(str(i) + "--" + str(df_train.iloc[i,1:].sum()))
df_train = df_train.drop(ls) # drops 26 rows with 0 annotations
'''

ds_train_cmplt = df_train.values
X_train_cmplt = ds_train_cmplt[:,:feature_size]
Y_train_cmplt = ds_train_cmplt[:,feature_size:]



ds_train = train.values
X_train = ds_train[:,:feature_size]
Y_train = ds_train[:,feature_size:]

ds_validate = validate.values
X_validate = ds_validate[:,:feature_size]
Y_validate = ds_validate[:,feature_size:]

df_test = pd.read_csv("Reduced_MLDA_400_test-bp.csv",header=None)

'''
ls =[]
for i in range(df_test.shape[0]):
    if df_test.iloc[i,1:].sum()==0:
        ls.append(i)
        print(str(i) + "--" + str(df_test.iloc[i,1:].sum()))
df_test = df_test.drop(ls)
'''

ds_test = df_test.values
X_test = ds_test[:,:feature_size]
Y_test = ds_test[:,feature_size:]

# k fold
tList,precList,recallList,f1List = [],[],[],[]
from sklearn.model_selection import KFold
fold=5
kfold = KFold(n_splits=fold, shuffle=True)
cvscores = []
i=0
for train, test in kfold.split(X_train_cmplt, Y_train_cmplt):
    filepath="model_" + str(i)+".h5"
    checkpoint = ModelCheckpoint(filepath,monitor="val_loss" ,verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model = Sequential()
    model.add(Dense(600, input_dim=feature_size, activation='relu'))
    model.add(Dense(labels, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    m = model.fit(X_train_cmplt[train], Y_train_cmplt[train] , verbose=1,epochs=500,validation_split=0.2,callbacks=callbacks_list, batch_size=256, class_weight="balanced")
    model = load_model(filepath)
    thresholds = [0.1,0.14, 0.16, 0.18, 0.20,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.4,0.5,0.6]
    plist,rlist,flist = [],[],[]
    for t in thresholds:
        Y_predicted = model.predict(X_train_cmplt[test])
        Y_predicted[Y_predicted>= t] = 1
        Y_predicted[Y_predicted< t] = 0
        avgPrecision, avgRecall, avgF1Score, F1Score, hammingLoss = metrics(Y_predicted, Y_train_cmplt[test])
        plist.append(avgPrecision)
        rlist.append(avgRecall)
        flist.append(F1Score)
    
    li=[]
    for ix in range(len(thresholds)):
        if plist[ix]>rlist[ix]:
            li.append(ix)
            
    znew = np.argmax([flist[l] for l in li])
    maxIndex=li[znew]

    #maxIndex = np.argmax(flist)
    print("maxIndex:" + str(maxIndex))
    t = thresholds[maxIndex]
    tList.append(t)
    print("threshold:" + str(t))
    Y_predicted = model.predict(X_train_cmplt[test])
    Y_predicted[Y_predicted>= t] = 1
    Y_predicted[Y_predicted< t] = 0
    avgPrecision, avgRecall, avgF1Score, F1Score, hammingLoss = metrics(Y_predicted, Y_train_cmplt[test])
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
    
##### test on cross validation models... NOT GOOD as the training data has been reduced
#### in case of cross validation while splitting the data into train test folds    
''' 
precList_test,recallList_test,f1List_test =[],[],[]
thres = sum(tList)/fold
for i in range(fold):
    filepath="model_" + str(i)+".h5"
    model = load_model(filepath)
    Y_predicted = model.predict(X_test)
    Y_predicted[Y_predicted>= thres] = 1
    Y_predicted[Y_predicted< thres] = 0
    avgPrecision, avgRecall, avgF1Score, F1Score, hammingLoss = metrics(Y_predicted, Y_test)
    precList_test.append(avgPrecision)
    recallList_test.append(avgRecall)
    f1List_test.append(F1Score)
'''
'''  	
    #model.fit(X_train[train], Y_train[train], epochs=150, batch_size=10, verbose=0)
    # evaluate the model
	scores = model.evaluate(X_train[test], Y_train[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
'''
	
precList_test,recallList_test,f1List_test, tList_test =[],[],[],[]
thres_mean = np.mean(tList)
thres_median = np.median(tList)
thres_mode = stats.mode(tList)
thres =[thres_mean,thres_median,list(thres_mode[0])]
# adam, dense 100, epoch 100, batch 128

for i in range(fold):
    model = Sequential()
    model.add(Dense(600, input_dim=feature_size, activation='relu'))
    model.add(Dense(labels, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    filepath="modeltest_"+str(i)+".h5"
    checkpoint = ModelCheckpoint(filepath,monitor="val_loss" ,verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    m = model.fit(X_train, Y_train , epochs=500,callbacks=callbacks_list, batch_size=256  ,validation_data=(X_validate, Y_validate), class_weight="balanced")
    
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
        
        avgPrecision, avgRecall, avgF1Score, F1Score, hammingLoss = metrics(Y_predicted, Y_test)
        tList_test.append(threshold)
        precList_test.append(avgPrecision)
        recallList_test.append(avgRecall)
        f1List_test.append(F1Score)
'''
thresholds = [0.14, 0.16, 0.18, 0.20,0.22,0.24,0.26,0.28,0.3,0.32,0.34]
plist,rlist,flist = [],[],[]
for t in thresholds:
    Y_predicted = model.predict(X_validate)
    Y_predicted[Y_predicted>= t] = 1
    Y_predicted[Y_predicted< t] = 0
    avgPrecision, avgRecall, avgF1Score, F1Score, hammingLoss = metrics(Y_predicted, Y_validate)

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
plt.legend(['precision','recall','f1-score'], loc = 'upper right')
#plt.show()
plt.savefig("PRgraph.png")


Y_predicted_test = model.predict(X_test)

threshold = 0.23
Y_predicted_test[Y_predicted_test >= threshold] = 1
Y_predicted_test[Y_predicted_test < threshold] = 0

avgPrecision, avgRecall, avgF1Score, F1Score, hammingLoss = metrics(Y_predicted_test, Y_test)
'''
'''
thhh=[0,1,2,3,4,5,6]
x=[1,2,3,4,5,0,9]
y=[3,2,1,0,0.2,8,6]
z=[2,3,2.8,2.9,2.8,3,2.95]
li=[]
for i in range(7):
    if x[i]>y[i]:
        li.append(i)
        
znew = np.argmax([z[i] for i in li])
ind=li[znew]
thhh[ind]        
'''
