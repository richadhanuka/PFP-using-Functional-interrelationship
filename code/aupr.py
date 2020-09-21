from keras.models import load_model
import numpy as np
import pandas as pd

df_test = pd.read_csv("R:\phd_backup\Final-work-1\BP\pearson\graph\seqgraphMLP\\test-multimodal-bp.csv")
feature_size = 351+256
ds_test = df_test.values
X_test = ds_test[:,:feature_size]
Y_test = ds_test[:,feature_size:]

labels= 932
filepath = 'R:\phd_backup\Final-work-1\BP\pearson\graph\seqgraphMLP\modeltest_4.h5'


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
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        lr_probs[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], lr_probs[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    lr_probs.ravel())
average_precision["micro"] = average_precision_score(Y_test, lr_probs,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))