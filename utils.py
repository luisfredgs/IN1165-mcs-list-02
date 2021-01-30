from deslib.util.instance_hardness import kdn_score
from deslib.util.diversity import double_fault
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import numpy as np

def get_validation_data(X, y, threshold = 0.5, hardnesses='hard'):
    score, kdn_neighbors = kdn_score(X, y, k=5)
    if hardnesses == 'hard':
        indices = np.where(score>0.5)
    elif hardnesses == 'easy':
        indices = np.where(score<=0.5)
    else:
        # original data with all their instances
        indices = np.where(score>=0.0)

    return X[indices], y[indices]


def voting(X, pool_classifiers, pruned_indices):
    if pruned_indices is not None:
        preds = np.array([pool_classifiers.estimators_[id].predict(X) for id in pruned_indices])
    else:
        preds = np.array([estimator.predict(X) for estimator in pool_classifiers.estimators_])

    # also one can try: stats.mode(preds)[0][0]
    maj_votes = np.apply_along_axis(lambda x:
                        np.argmax(np.bincount(x,weights=None)),
                        axis=0,
                        arr=preds.astype('int'))
    return maj_votes

# Diversity

def diversity_matrix(pool_classifiers, X, y, pruned_indices):
    if pruned_indices is not None:
        preds = np.array([pool_classifiers.estimators_[id].predict(X) for id in pruned_indices])
    else:
        preds = np.array([estimator.predict(X) for estimator in pool_classifiers.estimators_])


    n = len(preds)
    div = np.zeros((n,n))    
    for i in range(n):
        div[i][i] = double_fault(y, preds[i], preds[i])  

    for i in range(n):
        for j in range(i+1,n):
            div[i][j] = div[j][i] = double_fault(y, preds[i], preds[j])
    return div

# Metrics

def get_accuracy_score(y, predictions):
    accuracy = accuracy_score(y, predictions)
    return accuracy

def get_f1_score(y_true, predictions):
    return f1_score(y_true, predictions, average='macro')

def get_g1_score(y, predictions, average):
	precision = precision_score(y, predictions, average=average)
	recall = recall_score(y, predictions, average=average)
	return sqrt(precision*recall)