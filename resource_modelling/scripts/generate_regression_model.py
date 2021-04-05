import os
import math
import sklearn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, max_error
from finn.util.gdrive import *

def clean_dataframe(df):
    #This function checks if every parameter configuration has the resource utilization from
    #FINN estimate, HLS and Vivado synthesis. Removes everything else.
    #When splitting by the "Resources from:" column (estimate, hls, synthesis),
    #all 3 dataframes should have the same number of rows.
    
    #It takes ~5 min for a dataframe with ~22k samples

    #get dataframe headers
    headers = list(df)

    #get all parameters and resource classes
    parameters = []
    res_classes = []
    separator = 0
    for s in headers:
        if s == "Resources from:":
            separator = 1
        elif separator:
            res_classes.append(s)
        else:
            parameters.append(s)
    
    #move finn_commit, vivado_version and vivado_build_no from res_classes to parameters
    temp = ['finn_commit', 'vivado_version', 'vivado_build_no']
    for elem in temp:
        parameters.append(elem)
        res_classes.remove(elem)

    #sort the dataframe and reset index
    df = df.sort_values(parameters)
    df = df.reset_index(drop=True)

    #get synthesis data
    df_synth = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]
    df_temp = df
    keep_rows_list = []

    print("Started cleaning the dataframe")

    #for each parameter configuration search for estimate, hls and synthesis rows
    for index_config, configuration in df_synth[parameters].iterrows():
        if index_config not in keep_rows_list:
            found_row_estimate = 0
            found_row_hls = 0
            found_row_synthesis = 0
            for index, row in df_temp.iterrows():
                if configuration.isin(row).all():
                    if row["Resources from:"] == "estimate":
                        found_row_estimate = 1
                        index_row_estimate = index
                    elif row["Resources from:"] == "hls":
                        found_row_hls = 1
                        index_row_hls = index
                    elif row["Resources from:"] == "synthesis":
                        found_row_synthesis = 1
                        index_row_synthesis = index
                if found_row_estimate and found_row_hls and found_row_synthesis:
                    break
            if found_row_estimate and found_row_hls and found_row_synthesis:
                keep_rows_list.append(index_row_estimate)
                keep_rows_list.append(index_row_hls)
                keep_rows_list.append(index_row_synthesis)
                df_temp = df_temp.drop(index = index_row_estimate)
                df_temp = df_temp.drop(index = index_row_hls)
                df_temp = df_temp.drop(index = index_row_synthesis)
    
    df = df[df.index.isin(keep_rows_list)]
    
    #sort the dataframe and reset index
    df = df.sort_values(parameters)
    df = df.reset_index(drop=True)
    
    print("Finished cleaning the dataframe")

    return df

def filter_dataframe(df, filtering_dict):
    #filtering_dict template: {parameter1: [equal_flag, value1], parameter2: [equal_flag, value2]}
    #Example1: {"act": [False, "DataType.BIPOLAR"], "mem_mode": [True, "decoupled"]}
    #Example2: {"idt": [True, "DataType.UINT14", "DataType.UINT30"], "ich": [True, "48", "80", "160", "320"]}
    #The dataframe will be filtered by columns parameter1 and parameter2.
    #If equal_flag is set True, the rows where parameter == value will be selected,
    #else if equal_flag is set False, the rows where parameter != value will be selected.

    print("Started filtering the dataframe")

    for key in filtering_dict:
        if filtering_dict[key][0]:
            df = df[df[key].astype(str).isin(filtering_dict[key][1:])]
        else:
            df = df[~df[key].astype(str).isin(filtering_dict[key][1:])]

    print("Finished filtering the dataframe")

    return df

def datatype_strip(x):
    if "DataType.UINT" in str(x):
        return int(x.strip("DataType.UINT"))
    elif "DataType.INT" in str(x):
        return int(x.strip("DataType.INT"))
    elif "DataType.FLOAT" in str(x):
        return int(x.strip("DataType.FLOAT"))
    elif x in ["DataType.BINARY", "DataType.BIPOLAR"]:
        return 1
    elif x == "DataType.TERNARY":
        return 2
    else:
        return x

def extract_features_and_target(df, features, target):

    #DataType Strip on features
    for feature in features:
        df[feature] = df[feature].apply(datatype_strip)

    #TODO replace these with filter_dataframe function
    #get hls and finn estimate data
    df_hls = df[df.apply(lambda r: r.str.contains('hls', case=False).any(), axis=1)]
    df_finn_estimate = df[df.apply(lambda r: r.str.contains('estimate', case=False).any(), axis=1)]
    #get synth data
    df = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]

    #import pdb; pdb.set_trace()

    #extract features (X, df - synth data)
    X = df.loc[:, features].values
    X_hls = df_hls.loc[:, features].values
    X_finn_estimate = df_finn_estimate.loc[:, features].values

    #extract target
    Y = df.loc[:, target].values
    Y_hls = df_hls.loc[:, target].values
    Y_finn_estimate = df_finn_estimate.loc[:, target].values

    #check if the estimate, hls and synth features dfs are identical
    assert (X == X_hls).all(), 'X_hls different from X'
    assert (X == X_finn_estimate).all(), 'X_finn_estimate different from X'

    return X, Y, X_hls, Y_hls, X_finn_estimate, Y_finn_estimate

def gridsearch_hyperparameters(X_train, Y_train):
    #Returns an SVR estimator with hyperparameters tuned and fit on training data
    #Method to compute C - Source: Cherkassky, Vladimir & Ma, Yunqian. (2002). Selection of Meta-parameters for Support Vector Regression. Artif Neural Netw ICANN. 2002. 687-693. 10.1007/3-540-46084-5_112. 

    #compute mean and std of training targets to compute C
    mean = Y_train.mean()
    std = Y_train.std()
    C=max(abs(mean + 3*std), abs(mean - 3*std))

    #search for the best SVR hyperparameters
    best_svr = GridSearchCV(
        estimator=SVR(max_iter=-1),
        param_grid={
            'kernel': ['poly', 'rbf', 'sigmoid'],
            'gamma': ['scale'],
            'C': [C/10, max(abs(mean + std), abs(mean - std)), max(abs(mean + 2*std), abs(mean - 2*std)), max(abs(mean + 3*std), abs(mean - 3*std)), max(abs(mean + 4*std), abs(mean - 4*std)), max(abs(mean + 5*std), abs(mean - 5*std)), C*10],
            'epsilon': [10, 20, 50, 100, 150, 200, 250, 500, 1000]
        },
        cv=10, scoring = 'r2', n_jobs = -1, refit = True, verbose = 2)

    best_svr = best_svr.fit(X_train, Y_train)

    print("Best hyperparameters set found on development set:", best_svr.best_params_)
    #import pdb; pdb.set_trace()
    return best_svr

def generate_regression_model(df, features, target, scaler_selection=1, test_set_size=0.3, train_test_split_random_seed=2021):
    
    X, Y, X_hls, Y_hls, X_finn_estimate, Y_finn_estimate = extract_features_and_target(df, features, target)
    
    #split the data into train/test data sets - 30% testing, 70% training
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = test_set_size, shuffle=True, random_state=train_test_split_random_seed)
    
    #same split for hls and estimate, random needs to be set to same int value
    X_train_hls, X_test_hls, Y_train_hls, Y_test_hls = model_selection.train_test_split(X_hls, Y_hls, test_size = 0.3, shuffle=True, random_state=train_test_split_random_seed)
    X_train_finn_estimate, X_test_finn_estimate, Y_train_finn_estimate, Y_test_finn_estimate = model_selection.train_test_split(X_finn_estimate, Y_finn_estimate, test_size = 0.3, shuffle=True, random_state=train_test_split_random_seed)

    #Feature Scaling - Normalization(scaler_selection=0)/Standardization(scaler_selection=1)
    if scaler_selection:
        scaler = StandardScaler().fit(X_train)
    else:
        scaler = MinMaxScaler().fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    svr_estimator = gridsearch_hyperparameters(X_train, Y_train)

    #cross-validation
    scores = cross_val_score(svr_estimator, X_train, Y_train, cv=10, scoring = 'r2')
    print('Cross_val_scores:', scores)
    print('Cross_val_scores mean:', scores.mean())
    print('Cross_val_scores std:', scores.std())

    return svr_estimator, scaler, X_test, Y_test, Y_test_hls, Y_test_finn_estimate

def apply_scaler(scaler, X):
    
    X = scaler.transform(X)    
    return X

def compute_metrics(estimator, X_test, Y_test, Y_hls, Y_finn_estimate):
    #Computes the R2 score of the estimator on test dataset (X_test, Y_test).
    #Computes the RMSE and Max Error between Y_test and Y_predicted/Y_hls/Y_finn_estimate.

    print('R2 score test set:', estimator.score(X_test, Y_test))
    
    Y_predicted = estimator.predict(X_test)
    
    Y_relative_error_pred = (abs(Y_predicted - Y_test)/Y_test) * 100
    Y_relative_error_hls = (abs(Y_hls - Y_test)/Y_test) * 100
    Y_relative_error_estimate = (abs(Y_finn_estimate - Y_test)/Y_test) * 100
    """
    print(Y_test)
    print(Y_predicted)
    print(Y_relative_error_pred)
    print(Y_hls)
    print(Y_relative_error_hls)
    
    #compute means
    df = pd.DataFrame()
    df['Y_test'] = Y_test
    df['Y_rel_pred'] = Y_relative_error_pred
    df['Y_rel_hls'] = Y_relative_error_hls
    df['Y_rel_est'] = Y_relative_error_estimate

    df = df.sort_values('Y_test')
    df = df.reset_index(drop=True)

    print('0-1000 mean svr:', df['Y_rel_pred'][0:39].mean())
    print('1000-6000 mean svr:', df['Y_rel_pred'][39:58].mean())
    print('6000-12000 mean svr:', df['Y_rel_pred'][58:62].mean())

    print('0-1000 mean hls:', df['Y_rel_hls'][0:39].mean())
    print('1000-6000 mean hls:', df['Y_rel_hls'][39:58].mean())
    print('6000-12000 mean hls:', df['Y_rel_hls'][58:62].mean())

    print('0-1000 mean est:', df['Y_rel_est'][0:39].mean())
    print('1000-6000 mean est:', df['Y_rel_est'][39:58].mean())
    print('6000-12000 mean est:', df['Y_rel_est'][58:62].mean())
    import pdb; pdb.set_trace()
    
    fig = plt.figure(figsize=(20, 11))
    ax = fig.gca()

    X_test = [col[1] for col in X_test]
    print(X_test)

    ax.scatter(Y_test, Y_relative_error_estimate, marker="o", s=200, facecolors='none', edgecolors='g', label='estimate')
    ax.scatter(Y_test, Y_relative_error_hls, marker="^", s=500, facecolors='none', edgecolors='m', label='hls')
    ax.scatter(Y_test, Y_relative_error_pred, marker="x", s=50, color='r', label='predicted')
    ax.set_xlabel("LUTs")
    ax.set_ylabel("Relative_error [%]")
    leg = ax.legend()
    ax.set_ylim([0,100])
    fig.savefig('../graphs/plot_rel_error.png', bbox_inches='tight')
    """

    print('SVR Root Mean Squared Error:', mean_squared_error(Y_test, Y_predicted, squared = False))
    print('SVR Max Error:', max_error(Y_test, Y_predicted))

    print('HLS Root Mean Squared Error:', mean_squared_error(Y_test, Y_hls, squared = False))
    print('HLS Max Error:', max_error(Y_test, Y_hls))

    print('FINN Estimate Root Mean Squared Error:', mean_squared_error(Y_test, Y_finn_estimate, squared = False))
    print('FINN Estimate Max Error:', max_error(Y_test, Y_finn_estimate))
