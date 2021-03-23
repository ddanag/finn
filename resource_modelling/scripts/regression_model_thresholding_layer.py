import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as sts
from finn.util.gdrive import *
import sklearn
import math
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, max_error
import os, csv

#USER DEFINES
#define the worksheet name from finn-resource-dashboard
worksheet_name = "Thresholding_layer_resources"
#define the worksheet name of the unseen test set
test_set_worksheet_name = "Thresholding_layer_resources_test_set"
#define the directory name where to save the graphs
directory_name = "Thresholding"
#choose mem_mode ['decoupled', 'const']
sel_mem_mode = 'decoupled'
#choose ram_style ['distributed', 'block'] #+'auto'?
sel_ram_style = 'distributed'
#choose act type ['bipolar', 'non-bipolar']
sel_act = 'bipolar'

#create the directory where to save the graphs
new_dir_path = "../graphs/%s" % directory_name
try:
    os.mkdir(new_dir_path)
except OSError:
    print ("Creation of the directory %s failed" % new_dir_path)
else:
    print ("Successfully created the directory %s" % new_dir_path)

def datatype_strip(x):
    if "DataType.UINT" in x:
        return int(x.strip("DataType.UINT"))
    elif "DataType.INT" in x:
        return int(x.strip("DataType.INT"))
    elif "DataType.FLOAT" in x:
        return int(x.strip("DataType.FLOAT"))
    elif x in ["DataType.BINARY", "DataType.BIPOLAR"]:
        return 1
    elif x == "DataType.TERNARY":
        return 2
    else:
        raise Exception("Unrecognized data type: %s" % self.name)

#get all records from the selected worksheet
list_of_dicts = get_records_from_resource_dashboard(worksheet_name)
# convert list of dicts to dataframe
df = pd.DataFrame(list_of_dicts)

#get all records from the test set worksheet
test_set_list_of_dicts = get_records_from_resource_dashboard(test_set_worksheet_name)
#convert to dataframe
df_unseen_test_set = pd.DataFrame(test_set_list_of_dicts)

#get fpga
fpga = df['FPGA'].iloc[0]

#Filter df data based on defined mem_mode, ram_style, act
df = df[df['mem_mode'].astype(str) == sel_mem_mode]
df_unseen_test_set = df_unseen_test_set[df_unseen_test_set['mem_mode'].astype(str) == sel_mem_mode]

df = df[df['ram_style'].astype(str) == sel_ram_style]
df_unseen_test_set = df_unseen_test_set[df_unseen_test_set['ram_style'].astype(str) == sel_ram_style]

if sel_act == 'bipolar':
    df = df[df['act'].astype(str) == 'DataType.BIPOLAR']
    df_unseen_test_set = df_unseen_test_set[df_unseen_test_set['act'].astype(str) == 'DataType.BIPOLAR']
elif sel_act == 'non-bipolar':
    df = df[df['act'].astype(str) != 'DataType.BIPOLAR']
    df_unseen_test_set = df_unseen_test_set[df_unseen_test_set['act'].astype(str) != 'DataType.BIPOLAR']
    #get the bitwidth of act
    df['act'] = df['act'].apply(datatype_strip)
    df_unseen_test_set['act'] = df_unseen_test_set['act'].apply(datatype_strip)

#get the bitwidth of idt
df['idt'] = df['idt'].apply(datatype_strip)
df_unseen_test_set['idt'] = df_unseen_test_set['idt'].apply(datatype_strip)

#get hls and finn estimate data
df_hls = df[df.apply(lambda r: r.str.contains('hls', case=False).any(), axis=1)]
df_estimate = df[df.apply(lambda r: r.str.contains('estimate', case=False).any(), axis=1)]
#get synth data
df = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]

#divide the unseen test set df
df_unseen_ts_hls = df_unseen_test_set[df_unseen_test_set.apply(lambda r: r.str.contains('hls', case=False).any(), axis=1)]
df_unseen_ts_estimate = df_unseen_test_set[df_unseen_test_set.apply(lambda r: r.str.contains('estimate', case=False).any(), axis=1)]
df_unseen_ts_synth = df_unseen_test_set[df_unseen_test_set.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]

#get dataframe headers
headers = list(df)

#get all parameters and resource classes
parameters = []
res_classes = []
separator = 0
for s in headers:
    if s == 'Resources from:':
        separator = 1
    elif separator:
        res_classes.append(s)
    else:
        parameters.append(s)

#remove tools details: FPGA, finn_commit, vivado_version, vivado_build_no
#remove timing except Delay
#remove mem_mode, ram_style, 'Resources from:'
columns_to_remove = ['mem_mode', 'ram_style', 'Resources from:', 'FPGA', 'finn_commit', 'vivado_version', 'vivado_build_no', 'TargetClockPeriod', 'EstimatedClockPeriod', 'TargetClockFrequency [MHz]', 'EstimatedClockFrequency [MHz]']
res_classes = [element for element in res_classes if element not in columns_to_remove]
parameters = [element for element in parameters if element not in columns_to_remove]

df = df.drop(columns_to_remove, axis=1)
df_hls = df_hls.drop(columns_to_remove, axis=1)
df_estimate = df_estimate.drop(columns_to_remove, axis=1)

df_unseen_ts_synth = df_unseen_ts_synth.drop(columns_to_remove, axis=1)
df_unseen_ts_hls = df_unseen_ts_hls.drop(columns_to_remove, axis=1)
df_unseen_ts_estimate = df_unseen_ts_estimate.drop(columns_to_remove, axis=1)

pd.set_option('display.max_columns', 500)
print(parameters)
print(res_classes)
print(df)

#sort the dataframes
df = df.sort_values(parameters)
df_hls = df_hls.sort_values(parameters)
df_estimate = df_estimate.sort_values(parameters)

df_unseen_ts_synth = df_unseen_ts_synth.sort_values(parameters)
df_unseen_ts_hls = df_unseen_ts_hls.sort_values(parameters)
df_unseen_ts_estimate = df_unseen_ts_estimate.sort_values(parameters)

def generate_models(res_class):

    if sel_act == 'non-bipolar':
        features = ['ich', 'pe', 'idt', 'act']
    else:
        features = ['ich', 'pe', 'idt']

    #extract features (unseen test set)
    X_unseen_ts_hls = df_unseen_ts_hls.loc[:, features].values
    X_unseen_ts_estimate = df_unseen_ts_estimate.loc[:, features].values
    X_unseen_ts_synth = df_unseen_ts_synth.loc[:, features].values

    #check if the estimate, hls and synth features dfs are identical
    assert (X_unseen_ts_synth == X_unseen_ts_hls).all(), 'X_unseen_ts_hls different from X_unseen_ts_synth'
    assert (X_unseen_ts_synth == X_unseen_ts_estimate).all(), 'X_unseen_ts_estimate different from X_unseen_ts_synth'

    #extract targets (unseen test set)
    Y_unseen_ts_hls = df_unseen_ts_hls.loc[:, res_class].values
    Y_unseen_ts_estimate = df_unseen_ts_estimate.loc[:, res_class].values
    Y_unseen_ts_synth = df_unseen_ts_synth.loc[:, res_class].values

    #extract features (X, df - synth data)
    X = df.loc[:, features].values
    X_hls = df_hls.loc[:, features].values
    X_estimate = df_estimate.loc[:, features].values

    #check if the estimate, hls and synth features dfs are identical
    assert (X == X_hls).all(), 'X_hls different from X'
    assert (X == X_estimate).all(), 'X_estimate different from X'

    #extract target
    Y = df.loc[:, res_class].values
    Y_hls = df_hls.loc[:, res_class].values
    Y_estimate = df_estimate.loc[:, res_class].values
    
    #split the data into train/test data sets - 30% testing, 70% training
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3, shuffle=True, random_state=2021)
    
    #same split for hls and estimate, random needs to be set to same int value
    X_train_hls, X_test_hls, Y_train_hls, Y_test_hls = model_selection.train_test_split(X_hls, Y_hls, test_size = 0.3, shuffle=True, random_state=2021)
    X_train_estimate, X_test_estimate, Y_train_estimate, Y_test_estimate= model_selection.train_test_split(X_estimate, Y_estimate, test_size = 0.3, shuffle=True, random_state=2021)

    #Feature Scaling - Normalization/Standardization
    #scaler = MinMaxScaler().fit(X_train)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_unseen_ts_synth = scaler.transform(X_unseen_ts_synth)
    X_unseen_ts_hls = scaler.transform(X_unseen_ts_hls)
    X_unseen_ts_estimate = scaler.transform(X_unseen_ts_estimate)

    #linear regression
    linear_reg_model = LinearRegression()
    linear_reg_model = linear_reg_model.fit(X_train, Y_train)
    Y_predict_linear = linear_reg_model.predict(X_test)
    score_linear = linear_reg_model.score(X_test, Y_test)

    #SVR
    #compute mean and std of training targets to compute C
    mean = Y_train.mean()
    std = Y_train.std()
    C=max(abs(mean + 3*std), abs(mean - 3*std))

    #search for the best SVR hyperparameters
    gscv = GridSearchCV(
        estimator=SVR(max_iter=-1),
        param_grid={
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale'],
            'C': [C/10, max(abs(mean + std), abs(mean - std)), max(abs(mean + 2*std), abs(mean - 2*std)), max(abs(mean + 3*std), abs(mean - 3*std)), max(abs(mean + 4*std), abs(mean - 4*std)), max(abs(mean + 5*std), abs(mean - 5*std)), C*10],
            'epsilon': [10, 20, 50, 100, 150, 200, 250, 500, 1000]
        },
        cv=10, scoring = 'r2', n_jobs = -1, refit = True, verbose = 2)

    grid_result = gscv.fit(X_train, Y_train)

    print("Best parameters set found on development set:", grid_result.best_params_)
    
    #get best hyperparameters and define the model
    best_params = grid_result.best_params_
    best_svr = SVR(kernel=best_params["kernel"], C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"], max_iter=-1)

    #train the SVR model
    best_svr = best_svr.fit(X_train, Y_train)

    #cross-validation
    scores = cross_val_score(best_svr, X_train, Y_train, cv=10, scoring = 'r2')
    print('Cross_val_scores:', scores)
    print('Cross_val_scores mean:', scores.mean())
    print('Cross_val_scores std:', scores.std())

    print('R2 score test set:', best_svr.score(X_test, Y_test))
    Y_predict_svr = best_svr.predict(X_test)
    
    print('SVR Root Mean Squared Error:', mean_squared_error(Y_test, Y_predict_svr, squared = False))
    print('SVR Max Error:', max_error(Y_test, Y_predict_svr))

    print('HLS Root Mean Squared Error:', mean_squared_error(Y_test, Y_test_hls, squared = False))
    print('HLS Max Error:', max_error(Y_test, Y_test_hls))

    print('FINN Estimate Root Mean Squared Error:', mean_squared_error(Y_test, Y_test_estimate, squared = False))
    print('FINN Estimate Max Error:', max_error(Y_test, Y_test_estimate))

    ###unseen test set evaluation
    print('Unseen Test Set Evaluation:')
    Y_unseen_ts_predicted = best_svr.predict(X_unseen_ts_synth)
    print('R2 score test set:', best_svr.score(X_unseen_ts_synth, Y_unseen_ts_synth))

    print('SVR Root Mean Squared Error:', mean_squared_error(Y_unseen_ts_synth, Y_unseen_ts_predicted, squared = False))
    print('SVR Max Error:', max_error(Y_unseen_ts_synth, Y_unseen_ts_predicted))

    print('HLS Root Mean Squared Error:', mean_squared_error(Y_unseen_ts_synth, Y_unseen_ts_hls, squared = False))
    print('HLS Max Error:', max_error(Y_unseen_ts_synth, Y_unseen_ts_hls))

    print('FINN Estimate Root Mean Squared Error:', mean_squared_error(Y_unseen_ts_synth, Y_unseen_ts_estimate, squared = False))
    print('FINN Estimate Max Error:', max_error(Y_unseen_ts_synth, Y_unseen_ts_estimate))
    
    return X_test, Y_test, Y_predict_svr, best_svr.score(X_test, Y_test), Y_predict_linear, score_linear

def generate_graph_test_set(parameter, res_class):

    X_test, Y_test, Y_predict_svr, score_svr, Y_predict_linear, score_linear = generate_models(res_class)

    fig = plt.figure(figsize=(20, 11))
    ax = fig.gca()

    X_test = [col[1] for col in X_test]

    ax.scatter(Y_predict_linear, X_test, marker="o", s=200, facecolors='none', edgecolors='g', label='predicted_linear')
    ax.scatter(Y_predict_svr, X_test, marker="^", s=500, facecolors='none', edgecolors='m', label='predicted_svr')
    ax.scatter(Y_test, X_test, marker="x", s=50, color='r', label='synth')

    ax.set_xlabel("%s" % res_class)
    ax.set_ylabel("%s" % parameter)

    ax.set_title("%s vs %s (SVR_score = %s, linear_score = %s)" % (parameter, res_class, score_svr, score_linear))
    
    leg = ax.legend()
    
    fig.savefig('../graphs/%s/plot_%s_vs_%s_thresholding_%s_%s_%s.png' % (directory_name, parameter, res_class, sel_mem_mode, sel_ram_style, sel_act), bbox_inches='tight')

#for testing
parameters = ['pe']
res_classes = ['LUT']

for parameter in parameters:
    for res_class in res_classes:
        generate_graph_test_set(parameter, res_class)
        