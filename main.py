# The pruning algorithm is using the method 'Centralized Objection Maximization for Ensemble Pruning (COMEP)'\
#  of https://arxiv.org/pdf/1806.04899.pdf
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from deslib.util.diversity import double_fault
from sklearn.metrics import roc_auc_score
import seaborn as sns
import numpy as np
import time
from utils import *
from pruning import *
from datasets import *
import argparse

seed = 100000
np.random.seed(seed)
n_splits = 10

def run(args):
    
    hardnesses = args.hardnesses # easy, hard, all_instances    
    n_estimators = args.n_estimators
    base_learner = Perceptron(max_iter=100)
    print("Using %d classifiers." % n_estimators)
    if args.dataset == 'kc2':
        ds_name, X, Y = dataset_kc2()
    else:
        ds_name, X, Y = dataset_pc1()

    if args.pruning == True:
        pruning = True
        label_pruning_save = 'pruned'
        print("Classifiers will be pruned...")
    else:
        print("Training ensemble with no pruning...")
        pruning = False
        label_pruning_save = 'no_pruned'
    
    nb_pru = args.nb_pru

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    pool_classifiers = BaggingClassifier(base_estimator=base_learner, n_estimators=n_estimators, random_state=seed)
    scores = list()

    diversity_matrices = []
    results = {'accuracy': [], 'f1_score': [], 'g1_score': [], 'roc_auc':[], 'time_to_pruning': []}

    def train(train_index, test_index):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]        

        # train classifiers
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                            test_size=0.2, random_state=seed)
        pool_classifiers.fit(X_train, y_train)    
        
        Tc = 0.0
        if pruning == True:
            
            # Pruning classifiers    
            print("Pruning classifiers using COMEP...")            

            validation_data, validation_labels = get_validation_data(X_valid, y_valid, 0.5, hardnesses=hardnesses)        
            y_insp = [i.predict(validation_data).tolist() for i in pool_classifiers.estimators_]
            lam = 0.5
            #nb_pru = 5
            m = 4
            rho = nb_pru / n_estimators           

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

    def plot_diversity(df, save_to, annot=True):
        f, ax = plt.subplots(figsize=(12, 9))
        mask = np.triu(np.ones_like(df, dtype=bool))
        cmap = sns.cm.rocket_r
        sns.heatmap(df, fmt='g', square=True, annot=annot, linewidths=.0, cmap=cmap,
                    cbar_kws={"shrink": .5}, robust=True, mask=mask,
                    cbar=False)
        plt.savefig(save_to)

    if len(output) == n_splits:
        print(len(output))
        diversity_matrices = [out['diversity'] for out in output]
        results['accuracy'] = [out['acc'] for out in output]
        results['f1_score'] = [out['f1'] for out in output]
        results['g1_score'] = [out['g1'] for out in output]
        results['roc_auc'] = [out['roc'] for out in output]
        results['time_to_pruning'] = [out['time_to_pruning'] for out in output]

        # Results
        df_diversity = pd.DataFrame(np.mean(diversity_matrices, axis=0))
        df_diversity.to_csv("results/%s_diversity_matrix_%s_%s.csv" % (ds_name, hardnesses, label_pruning_save), index=False)

        
        #plot_diversity(df_diversity.round(2), "results/%s_diversity_matrix_%s_%s.pdf" % (ds_name, hardnesses, label_pruning_save))

        metric_results = pd.DataFrame(results)
        metric_results.to_csv("results/%s_summary_metrics_all_folds_%s_%s.csv" % (ds_name, hardnesses, label_pruning_save), index=False)

        print(metric_results.mean())
        metric_results.mean().to_csv("results/%s_summary_metrics_average_folds_%s_%s.csv" % (ds_name, hardnesses, label_pruning_save))
        

if __name__ == '__main__':        

    parser = argparse.ArgumentParser(description='Pruning classifiers')

    parser.add_argument('--hardnesses', dest='hardnesses',
                    default='all_instances', help='Instance hardness of validation set')

    parser.add_argument('--pruning', dest='pruning',
                    action='store_true', help='When pruning must be performed')
    
    parser.add_argument('--no-pruning', dest='pruning',
                    action='store_false', help='When pruning must be performed')
    
    parser.add_argument('--dataset', dest='dataset',
                    default="kc2", help='Dataset')
    
    parser.add_argument('--n_estimators', dest='n_estimators',
                    default=100, type=int, help='Number of base classifiers')

    parser.add_argument('--nb_pru', dest='nb_pru',
                    default=10, type=int, help='The size of the pruned sub-ensemble')
    
    args = parser.parse_args()

    run(args)
