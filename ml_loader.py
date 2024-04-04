import pickle
import pandas as pd
from pathlib import Path

def ctree_predict(xdf):
    modelPath = Path.cwd().joinpath('mdl')
    clf_mdl_name = modelPath.joinpath('clf.mdl')
    clf_mdl = pickle.load(open(clf_mdl_name,'rb'))
    clf_pred = clf_mdl.predict(xdf.values)
    return clf_pred

def RF_predict(xdf):
    modelPath = Path.cwd().joinpath('mdl')
    rf_mdl_name = modelPath.joinpath('rf.mdl')
    rf_mdl = pickle.load(open(rf_mdl_name,'rb'))
    rf_pred = rf_mdl.predict(xdf.values)
    return rf_pred

def SVM_predict(xdf):
    modelPath = Path.cwd().joinpath('mdl')
    SVM_mdl_name = modelPath.joinpath('svm.mdl')
    SVM_mdl = pickle.load(open(SVM_mdl_name,'rb'))
    SVM_pred = SVM_mdl.predict(xdf.values)
    return SVM_pred

def MLP_predict(xdf):
    modelPath = Path.cwd().joinpath('mdl')
    MLP_mdl_name = modelPath.joinpath('mlp.mdl')
    MLP_mdl = pickle.load(open(MLP_mdl_name,'rb'))
    MLP_pred = MLP_mdl.predict(xdf.values)
    return MLP_pred