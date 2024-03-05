import pickle
import pandas as pd

collecteddata = []

rf_mdl_name = ''
clf_mdl_name = ''
mlp_mdl_name = ''
svm_mdl_name = ''

rf_mdl = pickle.load(open(rf_mdl_name))
clf_mdl = pickle.load(open(clf_mdl_name))
mlp_mdl = pickle.load(open(mlp_mdl_name))
svm_mdl = pickle.load(open(svm_mdl_name))

rf_predicted = rf_mdl.predict(collecteddata);

