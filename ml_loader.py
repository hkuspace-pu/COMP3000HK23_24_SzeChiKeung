import pickle
import pandas as pd
from pathlib import Path

#modelPath = Path.cwd().joinpath('model')

# rf_mdl_name = modelPath.joinpath('rf.mdl')
# clf_mdl_name = modelPath.joinpath('clf.mdl')
# mlp_mdl_name = modelPath.joinpath('mlp.mdl')
# # svm_mdl_name = modelPath.joinpath('svm.mdl')

# rf_mdl = pickle.load(open(rf_mdl_name,'rb'))
# clf_mdl = pickle.load(open(clf_mdl_name,'rb'))
# mlp_mdl = pickle.load(open(mlp_mdl_name,'rb'))
# # svm_mdl = pickle.load(open(svm_mdl_name,'rb'))

# path = Path.cwd().joinpath('malicious-features').joinpath('features_1k.csv')
# data_1 = pd.read_csv(path, encoding = "ISO-8859-1")

# # Split the data into input and output variables
# x = data_1.iloc[1:2, 0:22].values   # Input features

# rf_pred = rf_mdl.predict(x)
# clf_pred = clf_mdl.predict(x)
# mlp_pred = mlp_mdl.predict(x)

# print(rf_pred,clf_pred,mlp_pred)

def ctree_predict(xdf):
    modelPath = Path.cwd().joinpath('mdl')
    clf_mdl_name = modelPath.joinpath('clf.mdl')
    clf_mdl = pickle.load(open(clf_mdl_name,'rb'))
    clf_pred = clf_mdl.predict(xdf.values)
    return clf_pred