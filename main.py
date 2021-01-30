from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold
from deslib.util.diversity import double_fault
from sklearn.metrics import roc_auc_score
import seaborn as sns
import numpy as np
import time

from utils import *
from pruning import *
from datasets import *

pruning = False
hardnesses = 'all_instances' # easy, all_instances
n_estimators = 10

base_learner = Perceptron(max_iter=100)
seed = 100000
np.random.seed(seed)

ds_name, X, Y = dataset_pc1()

if pruning == True:
    label_pruning_save = 'pruned'
else:
    label_pruning_save = 'no_pruned'


skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
pool_classifiers = BaggingClassifier(base_estimator=base_learner, n_estimators=n_estimators)
scores = list()

diversity_matrices = []
results = {'accuracy': [], 'f1_score': [], 'g1_score': [], 'roc_auc':[], 'time_to_pruning': []}

def train(train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Instance Hardness
    validation_data, validation_labels = get_validation_data(X_train, y_train, 0.5, hardnesses=hardnesses)        

    # train classifiers
    pool_classifiers.fit(X_train, y_train)    

    # Pruning classifiers    
    # https://arxiv.org/pdf/1806.04899.pdf
    

    y_insp = [i.predict(validation_data).tolist() for i in pool_classifiers.estimators_]
    
    lam = 0.5
    nb_pru = 5
    m = 4
    rho = nb_pru / n_estimators
    Tc = 0.0
    if pruning == True:
        # we must now run pruning using the method 'Centralized Objection Maximization for Ensemble Pruning (COMEP)'
        since = time.time()
        Pc = COMEP_Pruning(np.array(y_insp).T.tolist(), nb_pru, validation_labels, lam)
        Tc = time.time() - since
        print("{:5s}: {:.4f}s, get {}".format('Pruning', Tc, Pc))
        time_to_pruning = "{:5s}: {:.4f}s, get {}".format('Pruning', Tc, Pc)
            
        maj_votes = voting(X_test, pool_classifiers, pruned_indices=Pc)

        # Diversity
        diversity = diversity_matrix(pool_classifiers, X_test, y_test, pruned_indices=Pc)
    else:
        maj_votes = voting(X_test, pool_classifiers, pruned_indices=None)
        # Diversity
        diversity = diversity_matrix(pool_classifiers, X_test, y_test, pruned_indices=None)        
    
    
    acc = get_accuracy_score(y_test, maj_votes)
    g1 = get_g1_score(y_test, maj_votes, average='macro')
    f1 = get_f1_score(y_test, maj_votes)
    roc = roc_auc_score(y_test, maj_votes, average='macro')

    return dict(f1=f1, g1=g1, 
            acc=acc,
            roc=roc, 
            diversity=diversity,
            time_to_pruning=Tc)    

output = Parallel(n_jobs=-1, verbose=100, pre_dispatch='1.5*n_jobs')(
    delayed(train)(train_index, test_index) for train_index, test_index in skf.split(X, Y))


diversity_matrices = [out['diversity'] for out in output]
results['accuracy'] = [out['acc'] for out in output]
results['f1_score'] = [out['f1'] for out in output]
results['g1_score'] = [out['g1'] for out in output]
results['roc_auc'] = [out['roc'] for out in output]
results['time_to_pruning'] = [out['time_to_pruning'] for out in output]

# Results
df_diversity = pd.DataFrame(np.mean(diversity_matrices, axis=0))
df_diversity.to_csv("results/%s_diversity_matrix_%s_%s.csv" % (ds_name, hardnesses, label_pruning_save), index=False)

with plt.style.context('ggplot'):
    sns.heatmap(df_diversity.round(2), annot=True, fmt="g", cmap='viridis', xticklabels=True, yticklabels=True, annot_kws={'size':10})
    plt.savefig("results/%s_diversity_matrix_%s_%s.pdf" % (ds_name, hardnesses, label_pruning_save))
    #plt.show()


metric_results = pd.DataFrame(results)
metric_results.to_csv("results/%s_summary_metrics_%s_%s.csv" % (ds_name, hardnesses, label_pruning_save), index=False)

print(metric_results.mean())