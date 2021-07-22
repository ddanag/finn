import os
import math
import sklearn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as sts
from numpy import inf
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_percentage_error
from finn.util.gdrive import *
import csv

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
    #temp = ['finn_commit', 'vivado_version', 'vivado_build_no']
    #ignore finn_commit
    temp = ['vivado_version', 'vivado_build_no']
    res_classes.remove('finn_commit')
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
                #if configuration.isin(row).all(): -there's a problem with this returns true when it shouldn't    
                flag = True
                for key in configuration.keys():
                    if configuration[key] != row[key]:     
                        flag = False
                        break
                if flag == True:        
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

def remove_fully_unfolded_configs(df):

    drop_index_list = []

    for index, row in df.iterrows():
        if row["mh"] == row["pe"] and row["mw"] == row["simd"]:
            drop_index_list.append(index)
   
    df = df.drop(drop_index_list)
    df = df.reset_index(drop=True)

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
    elif x == "None":
        return 0
    else:
        return x

def extract_features_and_target(df, features, target):

    #DataType Strip on features
    for feature in features:
        df[feature] = df[feature].apply(datatype_strip)

        #TODO save both encoders
        label_encoder = LabelEncoder()
        if feature == 'ram_style' or feature == 'mem_mode': 
            df[feature] = label_encoder.fit_transform(df[feature])
    
    df['LUTRAM'] = df['LUTRAM'].replace('-', 0)
    
    #get hls and finn estimate data
    df_hls = df[df.apply(lambda r: r.str.contains('hls', case=False).any(), axis=1)]
    df_finn_estimate = df[df.apply(lambda r: r.str.contains('estimate', case=False).any(), axis=1)]
    #get synth data
    df = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]
    
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
    
    return X, Y, X_hls, Y_hls, X_finn_estimate, Y_finn_estimate, label_encoder

def gridsearch_hyperparameters(X_train, Y_train, epsilon_grid):
    #Returns an SVR estimator with hyperparameters tuned and fit on training data
    #Method to compute C - Source: Cherkassky, Vladimir & Ma, Yunqian. (2002). Selection of Meta-parameters for Support Vector Regression. Artif Neural Netw ICANN. 2002. 687-693. 10.1007/3-540-46084-5_112. 

    #compute mean and std of training targets to compute C
    mean = Y_train.mean()
    std = Y_train.std()
    C=max(abs(mean + 3*std), abs(mean - 3*std))

    #search for the best SVR hyperparameters
    gscv_svr = GridSearchCV(
        estimator=SVR(max_iter=2000000), #2000000 to stop when there is a convergence problem, #-1 default
        param_grid={
            #'kernel': ['poly', 'rbf', 'sigmoid'],
            'kernel': ['rbf'],
            'gamma': ['scale'],
            #'C': [max(abs(mean + std), abs(mean - std)), max(abs(mean + 2*std), abs(mean - 2*std)), max(abs(mean + 3*std), abs(mean - 3*std)), max(abs(mean + 4*std), abs(mean - 4*std)), max(abs(mean + 5*std), abs(mean - 5*std))],
            'C': [C/10, max(abs(mean + std), abs(mean - std)), max(abs(mean + 2*std), abs(mean - 2*std)), max(abs(mean + 3*std), abs(mean - 3*std)), max(abs(mean + 4*std), abs(mean - 4*std)), max(abs(mean + 5*std), abs(mean - 5*std)), C*10],
            'epsilon': epsilon_grid
        },
        cv=10, scoring = 'r2'  , n_jobs = -1, refit = True, verbose = 2)
    #some alternatives for scoring: ['r2', 'max_error', 'neg_mean_squared_error', 'neg_mean_absolute_percentage_error']
    gscv_svr = gscv_svr.fit(X_train, Y_train)
    best_svr = gscv_svr.best_estimator_
    print("Best hyperparameters set found on development set:", gscv_svr.best_params_)
    return best_svr

def generate_regression_model(df, features, target, feature_scaler_selection=1, test_set_size=0.3, train_test_split_random_seed=2021, epsilon_dict={}, target_scaler = None):
    
    #target_scaler:   0 - log
    #                 1 - (synth-finn_estimate)
    #                 None    

    X, Y, X_hls, Y_hls, X_finn_estimate, Y_finn_estimate, label_encoder = extract_features_and_target(df, features, target)

    #split the data into train/test data sets - 30% testing, 70% training
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = test_set_size, shuffle=True, random_state=train_test_split_random_seed)
    X_test_before_processing = X_test
    
    #same split for hls and estimate, random needs to be set to same int value
    X_train_hls, X_test_hls, Y_train_hls, Y_test_hls = model_selection.train_test_split(X_hls, Y_hls, test_size = test_set_size, shuffle=True, random_state=train_test_split_random_seed)
    X_train_finn_estimate, X_test_finn_estimate, Y_train_finn_estimate, Y_test_finn_estimate = model_selection.train_test_split(X_finn_estimate, Y_finn_estimate, test_size = test_set_size, shuffle=True, random_state=train_test_split_random_seed)

    #Feature Scaling - Normalization(feature_scaler_selection=0)/Standardization(feature_scaler_selection=1)
    if feature_scaler_selection:
        feature_scaler = StandardScaler().fit(X_train)
    else:
        feature_scaler = MinMaxScaler().fit(X_train)
    
    ##delete this when the problem with saving and restoring the models is fixed
    X_train_before = X_train

    X_train = feature_scaler.transform(X_train)
    X_test = feature_scaler.transform(X_test)
   
    #Target Scaling/Preprocessing
    if target_scaler == 1:
        Y_train = Y_train - Y_train_finn_estimate
    elif target_scaler == 0:
        Y_train = np.log(Y_train)
        #replace -inf results from log(0) with 0
        Y_train = np.asarray([0 if x == -inf else x for x in Y_train])

    if len(epsilon_dict) == 0:
        if target_scaler != None:
            epsilon_dict =  {   "LUT" : [0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
                                "LUTRAM" : [0.001, 0.01, 0.1, 1, 2, 5, 10],
                                "FF" : [0.001, 0.01, 0.1, 1, 2, 5],
                                "SRL" : [],
                                "DSP" : [],
                                "Total_BRAM_18K": [0.001, 0.005, 0.01, 0.1, 0.25, 0.5, 1, 1.25, 1.5],
                                "URAM": [0.01, 0.1, 1, 2, 5, 10],
                                "Carry": [0.001, 0.01, 0.1, 1, 2, 5, 10],
                                "Delay": [0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
                            }
        else:
            epsilon_dict =  {   "LUT" : [1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 500, 1000],
                                "LUTRAM" : [1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 500, 1000],
                                "FF" : [1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 500],
                                "SRL" : [],
                                "DSP" : [],
                                "Total_BRAM_18K": [1, 5, 10, 15, 20],
                                "URAM": [1, 2, 5, 10, 15, 20],
                                "Carry": [1, 2, 5, 10, 15, 20, 50, 100],
                                "Delay": [0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
                            }

    svr_estimator = gridsearch_hyperparameters(X_train, Y_train, epsilon_dict[target])

    #cross-validation
    scores = cross_val_score(svr_estimator, X_train, Y_train, cv=10, scoring = 'r2')
    
    print('Cross_val_scores:', scores)
    print('Cross_val_scores mean:', scores.mean())
    print('Cross_val_scores std:', scores.std())

    return svr_estimator, feature_scaler, target_scaler, label_encoder, X_train_before, X_train, Y_train, X_test, Y_test, Y_test_hls, Y_test_finn_estimate, X_test_before_processing

def apply_feature_scaler(feature_scaler, X):
    
    X = feature_scaler.transform(X)    
    return X

def compute_metrics(estimator, target_scaler, X_test, Y_test, Y_hls, Y_finn_estimate):
    #Computes the R2 score of the estimator on test dataset (X_test, Y_test).
    #Computes the RMSE and Max Error between Y_test and Y_predicted/Y_hls/Y_finn_estimate.

    if target_scaler == 1:
        Y_test= Y_test - Y_finn_estimate
        print('R2 score test set:', estimator.score(X_test, Y_test))
        Y_test = Y_test + Y_finn_estimate
        Y_predicted = estimator.predict(X_test) + Y_finn_estimate
    elif target_scaler == 0:
        Y_test = np.log(Y_test)
        #replace -inf results from log(0) with 0 to compute score
        Y_test_score = np.asarray([0 if x == -inf else x for x in Y_test])
        print('R2 score test set:', estimator.score(X_test, Y_test_score))
        Y_test = np.exp(Y_test)
        Y_predicted = np.exp(estimator.predict(X_test))
    else:
        print('R2 score test set:', estimator.score(X_test, Y_test))
        Y_predicted = estimator.predict(X_test)

    print('SVR Root Mean Squared Error:', mean_squared_error(Y_test, Y_predicted, squared = False))
    print('SVR Max Error:', max_error(Y_test, Y_predicted))
    print('SVR Mean Abs Percentage Error:', mean_absolute_percentage_error(Y_test, Y_predicted))

    print('HLS Root Mean Squared Error:', mean_squared_error(Y_test, Y_hls, squared = False))
    print('HLS Max Error:', max_error(Y_test, Y_hls))
    print('HLS Mean Abs Percentage Error:', mean_absolute_percentage_error(Y_test, Y_hls))
    
    print('FINN Estimate Root Mean Squared Error:', mean_squared_error(Y_test, Y_finn_estimate, squared = False))
    print('FINN Estimate Max Error:', max_error(Y_test, Y_finn_estimate))
    print('FINN Mean Abs Percentage Error:', mean_absolute_percentage_error(Y_test, Y_finn_estimate))

def plot_relative_error_graph(estimator, target_scaler, X_test, Y_test, Y_hls, Y_finn_estimate, target, directory_name):

    if target_scaler == 1:
        Y_predicted = estimator.predict(X_test) + Y_finn_estimate
    elif target_scaler == 0:
        Y_predicted = np.exp(estimator.predict(X_test))
    else:
        Y_predicted = estimator.predict(X_test)

    #Solution to "zero division error" for relative error computation - using abs(x - x_true)/(1 + abs(x_true))
    Y_test_denominator = np.asarray([(abs(x) + 1) if x == 0 else x for x in Y_test])
    
    #Y_relative_error_pred = (abs(Y_predicted - Y_test)/Y_test_denominator) * 100
    #Y_relative_error_hls = (abs(Y_hls - Y_test)/Y_test_denominator) * 100
    #Y_relative_error_estimate = (abs(Y_finn_estimate - Y_test)/Y_test_denominator) * 100
    
    Y_relative_error_pred = ((Y_predicted - Y_test)/Y_test_denominator) * 100
    Y_relative_error_hls = ((Y_hls - Y_test)/Y_test_denominator) * 100
    Y_relative_error_estimate = ((Y_finn_estimate - Y_test)/Y_test_denominator) * 100
    
    #compute mean relative error
    df = pd.DataFrame()
    df['Y_test'] = Y_test
    df['Y_rel_pred'] = Y_relative_error_pred
    df['Y_rel_hls'] = Y_relative_error_hls
    df['Y_rel_est'] = Y_relative_error_estimate

    df = df.sort_values('Y_test')
    df = df.reset_index(drop=True)
    if target == "LUT" or target == "LUTRAM" or target == "FF":
        x = 1000
        y = 10000
    else:
        x = 10
        y = 100

    df_s = df[df["Y_test"] < x]
    df_m = df[(df["Y_test"] > x) & (df["Y_test"] < y)]
    df_l = df[df["Y_test"] > y]  
    
    print("SVR Mean relative error:")
    print("0-%s %s" %(x, target), df_s["Y_rel_pred"].mean())
    print("%s-%s %s" %(x, y, target), df_m["Y_rel_pred"].mean())
    print("> %s %s" %(y, target), df_l["Y_rel_pred"].mean())

    print("HLS Mean relative error:")
    print("0-%s %s" %(x, target), df_s["Y_rel_hls"].mean())
    print("%s-%s %s" %(x, y, target), df_m["Y_rel_hls"].mean())
    print("> %s %s" %(y, target), df_l["Y_rel_hls"].mean())

    print("FINN Mean relative error:")
    print("0-%s %s" %(x, target), df_s["Y_rel_est"].mean())
    print("%s-%s %s" %(x, y, target), df_m["Y_rel_est"].mean())
    print("> %s %s" %(y, target), df_l["Y_rel_est"].mean())

    fig = plt.figure(figsize=(20, 11))
    ax = fig.gca()

    X_test = [col[1] for col in X_test]

    ax.scatter(Y_test, Y_relative_error_estimate, marker="o", s=200, facecolors='none', edgecolors='g', label='estimate')
    ax.scatter(Y_test, Y_relative_error_hls, marker="^", s=500, facecolors='none', edgecolors='m', label='hls')
    ax.scatter(Y_test, Y_relative_error_pred, marker="x", s=50, color='r', label='predicted')
    ax.set_xlabel("%s" % target)
    ax.set_ylabel("Relative_error [%]")
    leg = ax.legend()
    fig.savefig('../graphs/%s/plot_rel_error_vs_%s.png' % (directory_name, target), bbox_inches='tight')
    
    ax.set_xscale('log')
    fig.savefig('../graphs/%s/plot_rel_error_vs_%s_log.png' % (directory_name, target), bbox_inches='tight')

    ax.set_yscale('log')
    fig.savefig('../graphs/%s/plot_rel_error_vs_%s_log-log.png' % (directory_name, target), bbox_inches='tight')
   
    ax.set_yscale('linear')
    ax.set_xscale('linear')
    ax.set_ylim([0,100])
    fig.savefig('../graphs/%s/plot_rel_error_vs_%s_zoom.png' % (directory_name, target), bbox_inches='tight')

    #boxplot
    df_boxplot = pd.DataFrame()
    df_boxplot["finn_error_overest"] = pd.Series(Y_relative_error_estimate[Y_relative_error_estimate > 0])
    df_boxplot["finn_error_underest"] =  pd.Series(Y_relative_error_estimate[Y_relative_error_estimate < 0])
    
    df_boxplot["hls_error_overest"] = pd.Series(Y_relative_error_hls[Y_relative_error_hls > 0])
    df_boxplot["hls_error_underest"] =  pd.Series(Y_relative_error_hls[Y_relative_error_hls < 0])
    
    df_boxplot["svr_error_overest"] = pd.Series(Y_relative_error_pred[Y_relative_error_pred > 0])
    df_boxplot["svr_error_underest"] =  pd.Series(Y_relative_error_pred[Y_relative_error_pred < 0])
    
    print('finn overest = %s out of %s' %(len(pd.Series(Y_relative_error_estimate[Y_relative_error_estimate > 0])), len(Y_test)))
    print('finn underest = %s out of %s' %(len(pd.Series(Y_relative_error_estimate[Y_relative_error_estimate < 0])), len(Y_test)))

    print('hls overest = %s out of %s' %(len(pd.Series(Y_relative_error_hls[Y_relative_error_hls > 0])), len(Y_test)))
    print('hls underest = %s out of %s' %(len(pd.Series(Y_relative_error_hls[Y_relative_error_hls < 0])), len(Y_test)))

    print('svr overest = %s out of %s' %(len(pd.Series(Y_relative_error_pred[Y_relative_error_pred > 0])), len(Y_test)))
    print('svr underest = %s out of %s' %(len(pd.Series(Y_relative_error_pred[Y_relative_error_pred < 0])), len(Y_test)))

    fig2 = plt.figure(figsize=(20, 11))
    boxplot = df_boxplot.boxplot(showfliers=True, patch_artist=True)
    fig2.savefig('../graphs/%s/plot_rel_error_boxplot_%s_second.png' % (directory_name, target), bbox_inches='tight')


def plot_pareto_frontier_graph(estimator, target_scaler, X_test, Y_test, Y_hls, Y_finn_estimate, target, directory_name):
    
    if target_scaler == 1:
        Y_predicted = estimator.predict(X_test) + Y_finn_estimate
    elif target_scaler == 0:
        Y_predicted = np.exp(estimator.predict(X_test))
    else:
        Y_predicted = estimator.predict(X_test)

    df = pd.DataFrame()
    df['Y_test'] = Y_test
    df['Y_predicted'] = Y_predicted
    df['Y_hls'] = Y_hls
    df['Y_finn_estimate'] = Y_finn_estimate

    df = df.sort_values('Y_test')
    df = df.reset_index(drop=True)

    for index, row in df.iterrows():
        pred = abs(row['Y_test'] - row['Y_predicted'])
        hls = abs(row['Y_test'] - row['Y_hls'])
        finn = abs(row['Y_test'] - row['Y_finn_estimate'])
        if (pred < hls) and (pred < finn):
            df.at[index, 'Y_hls'] = -1000
            df.at[index, 'Y_finn_estimate'] = -1000
        elif (hls < pred) and (hls < finn):
            df.at[index, 'Y_predicted'] = -1000
            df.at[index, 'Y_finn_estimate'] = -1000
        elif (finn < pred) and (finn < hls):
            df.at[index, 'Y_predicted'] = -1000
            df.at[index, 'Y_hls'] = -1000

    fig = plt.figure(figsize=(20, 11))
    ax = fig.gca()

    ax.scatter(df['Y_test'], df['Y_finn_estimate'], marker="o", s=200, facecolors='none', edgecolors='g', label='estimate')
    ax.scatter(df['Y_test'], df['Y_hls'], marker="^", s=300, facecolors='none', edgecolors='m', label='hls')
    ax.scatter(df['Y_test'],  df['Y_predicted'], marker="x", s=50, color='r', label='predicted')
    ax.set_xlabel("%s" % target)
    ax.set_ylabel("Closest estimation")
    leg = ax.legend()
    
    ax.set_ylim(bottom=0)
    fig.savefig('../graphs/%s/plot_pareto_%s.png' % (directory_name, target), bbox_inches='tight')


def save_test_results_to_csv(svr_estimator, target_scaler, label_encoder, X_test_before_processing, X_test, Y_test, Y_hls, Y_finn_estimate, target, directory_name, features):
    #X_test - after processing

    filename = 'test_set_results_%s_%s_without_fu.csv'% (directory_name, target)
    filepath = "../graphs/%s/%s" % (directory_name, filename)

    df = pd.DataFrame(X_test_before_processing, columns = features)
    try:
        label_classes = label_encoder.classes_.tolist()
        df['mem_mode'] = df['mem_mode'].map({0:label_classes[0], 1:label_classes[1], 2:label_classes[2]})
    except:
        print("No label classes")

    df['LUT synth'] = Y_test
    df['LUT hls'] = Y_hls
    df['LUT finn'] = Y_finn_estimate

    if target_scaler == 1:
        Y_predicted = svr_estimator.predict(X_test) + Y_finn_estimate
    elif target_scaler == 0:
        Y_predicted = np.exp(svr_estimator.predict(X_test))
    else:
        Y_predicted = svr_estimator.predict(X_test)

    #Solution to "zero division error" for relative error computation - using abs(x - x_true)/(1 + abs(x_true))
    Y_test_denominator = np.asarray([(abs(x) + 1) if x == 0 else x for x in Y_test])
    
    Y_relative_error_pred = (abs(Y_predicted - Y_test)/Y_test_denominator) * 100
    Y_relative_error_hls = (abs(Y_hls - Y_test)/Y_test_denominator) * 100
    Y_relative_error_estimate = (abs(Y_finn_estimate - Y_test)/Y_test_denominator) * 100

    df['LUT svr'] = Y_predicted
    df['finn_rel_error'] = Y_relative_error_estimate
    df['svr_rel_error'] = Y_relative_error_pred
    df['finn/svr rel error'] = Y_relative_error_estimate/Y_relative_error_pred

    df.to_csv(filepath, index = False, header=True)

    #import pdb; pdb.set_trace()