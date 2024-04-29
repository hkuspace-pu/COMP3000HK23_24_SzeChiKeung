import pickle
import pandas as pd
import numpy
from pathlib import Path

def c45_predict(xdf):
    # modelPath = Path.cwd().joinpath('mdl')
    modelPath = Path.cwd().joinpath('testcase').joinpath('f_mdl')
    # clf_mdl_name = modelPath.joinpath('clf.mdl')
    c45_mdl_name = modelPath.joinpath('c45_f.mdl')
    c45_mdl = pickle.load(open(c45_mdl_name,'rb'))
    c45_pred = c45_mdl.predict(xdf.values)
    return numpy.array(c45_pred)

def ctree_predict(xdf):
    # modelPath = Path.cwd().joinpath('mdl')
    modelPath = Path.cwd().joinpath('testcase').joinpath('f_mdl')
    # clf_mdl_name = modelPath.joinpath('clf.mdl')
    clf_mdl_name = modelPath.joinpath('clf_f.mdl')
    clf_mdl = pickle.load(open(clf_mdl_name,'rb'))
    clf_pred = clf_mdl.predict(xdf.values)
    return clf_pred

def RF_predict(xdf):
    # modelPath = Path.cwd().joinpath('mdl')
    modelPath = Path.cwd().joinpath('testcase').joinpath('f_mdl')
    # rf_mdl_name = modelPath.joinpath('rf.mdl')
    rf_mdl_name = modelPath.joinpath('rf_f.mdl')
    rf_mdl = pickle.load(open(rf_mdl_name,'rb'))
    rf_pred = rf_mdl.predict(xdf.values)
    return rf_pred

def SVM_predict(xdf):
    modelPath = Path.cwd().joinpath('testcase').joinpath('1m_mdl')
    SVM_mdl_name = modelPath.joinpath('svm.mdl')
    SVM_mdl = pickle.load(open(SVM_mdl_name,'rb'))
    SVM_pred = SVM_mdl.predict(xdf.values)
    return SVM_pred

def MLP_predict(xdf):
    # modelPath = Path.cwd().joinpath('mdl')
    modelPath = Path.cwd().joinpath('testcase').joinpath('f_mdl')
    # MLP_mdl_name = modelPath.joinpath('mlp.mdl')
    MLP_mdl_name = modelPath.joinpath('mlp_f.mdl')
    MLP_mdl = pickle.load(open(MLP_mdl_name,'rb'))
    MLP_pred = MLP_mdl.predict(xdf.values)
    return MLP_pred