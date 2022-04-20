from cProfile import label
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
"""y_true = np.random.randint(low=0,high=5, size=100)
y_pred = np.random.randint(low=0,high=5, size=100)

labels = [x for x in range(0,14)]
print(labels)
cm = np.array(confusion_matrix(y_true, y_pred))

idx = 1

score = precision_score(y_true, y_pred, average='weighted', labels=labels)
# recall = recall_score(y_true, y_pred, average='weighted', labels=[idx])
# f1 = f1_score(y_true, y_pred, average='weighted', labels=[idx])
# score = precision_score(y_true, y_pred, average='weighted')
# recall = recall_score(y_true, y_pred, average='weighted')
# f1 = f1_score(y_true, y_pred, average='weighted')
print('precision: ', score)
# print('recall: ', recall)
# print('f1: ',f1)
print(cm)
print(cm + cm)

FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
print(FP, FN, TP, TN)
# precision = float(TP.sum() / (TP.sum()+FP.sum()))
# recall =  float(TP.sum() / (TP.sum()+FN.sum()))

precision = float(TP[idx] / (TP[idx]+FP[idx]))
recall =  float(TP[idx] / (TP[idx]+FN[idx]))
f1 = 2 * (precision*recall) / (precision+recall)
print('pre: ',precision)
print('recall: ', recall)
print('f1: ',f1);

WP = 0
for idx in range(0,5):
    precision = float(TP[idx] / (TP[idx]+FP[idx]))
    recall =  float(TP[idx] / (TP[idx]+FN[idx]))
    f1 = 2 * (precision*recall) / (precision+recall)

    WP += precision*(TP[idx]+FN[idx])
print(WP/100)"""


y_true = np.random.randint(low=0,high=8, size=20)
y_pred = np.random.randint(low=0,high=8, size=20)
cm = np.array(confusion_matrix(y_true, y_pred))
print(cm)
print(cm.shape)
print(y_true)
print(y_pred)

tmp = np.append(y_true, y_pred)
ele = np.unique(tmp)

for idx in range(0,10):
    if (idx not in ele):
        cm = np.insert(cm, idx, np.zeros(cm.shape[0]), axis=1)
        cm = np.insert(cm, idx, np.zeros(cm.shape[1]), axis=0)


print(cm)
print(cm.shape)