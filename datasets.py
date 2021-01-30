import pandas as pd
from sklearn import preprocessing
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

def dataset_kc2():
    """KC2/software defect prediction
    Data comes from McCabe and Halstead features extractors of source code.  These features were defined in the 70s in an attempt
    to objectively characterize code features that are associated with software quality.  The nature of association is under dispute.    
    """
    data = arff.loadarff("data/kc2.arff")
    df = pd.DataFrame(data[0])
    print(df.info())

    le = preprocessing.LabelEncoder()
    le.fit(df['problems'])

    X = df.drop(['problems'], axis=1).to_numpy()

    sc = StandardScaler()
    X = sc.fit_transform(X)
    Y = le.transform(df['problems']) 
    return 'kc2', X, Y

def dataset_pc1():
    """PC1 Software defect prediction"""

    data = arff.loadarff("data/pc1.arff")
    df = pd.DataFrame(data[0])
    print(df.info())

    le = preprocessing.LabelEncoder()
    le.fit(df['defects'])

    X = df.drop(['defects'], axis=1).to_numpy()

    sc = StandardScaler()
    X = sc.fit_transform(X)
    Y = le.transform(df['defects']) 
    return 'pc1', X, Y
