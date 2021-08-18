from finn.util.gdrive import *
from generate_regression_model import *
import json
import numpy as np
import math

import sklearn
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#USER DEFINES
#define the worksheet name from finn-resource-dashboard
worksheet_name = "FCLayer_resources"

#define the directory name where to save the graphs
directory_name = "FCLayer"

#get all records from the selected worksheet
list_of_dicts = get_records_from_resource_dashboard(worksheet_name)
# convert list of dicts to dataframe
df = pd.DataFrame(list_of_dicts)

#filter dataframe
#df = filter_dataframe(df, filtering_dict)

#remove the fully unfolded configurations - outliers
df = remove_fully_unfolded_configs(df, directory_name)

#Hint: clean the dataframe after filtering because cleaning 
#takes a while (~5 min for ~22k samples)
df = clean_dataframe(df)
df_initial = df

def plot_relative_error_graph(predictor, X_test, Y_test, Y_hls, Y_finn_estimate):

    Y_predicted = predictor.predict(X_test)

    Y_test = Y_test.astype(int)
    Y_predicted = Y_predicted.astype(int)
    Y_hls = Y_hls.astype(int)
    Y_finn_estimate = Y_finn_estimate.astype(int)

    #count how many times hls and prev finn made the correct decision
    error_finn_prev = abs(Y_finn_estimate - Y_test)
    error_hls = abs(Y_hls - Y_test)
    error_pred = abs(Y_predicted - Y_test)

    count_finn_prev = error_finn_prev.tolist().count(0)
    count_hls = error_hls.tolist().count(0)
    count_pred = error_pred.tolist().count(0)

    accuracy_finn_prev = count_finn_prev/len(X_test) * 100
    accuracy_hls = count_hls/len(X_test) * 100
    accuracy_pred = count_pred/len(X_test) * 100

    x_labels = ["HLS", "Previous FINN method", "DT Classifier"]
    y_values = [accuracy_hls, accuracy_finn_prev, accuracy_pred]

    fig = plt.figure(figsize=(8, 11))
    plt.bar(x_labels, y_values, color ='r', width = 0.2)
 
    plt.ylabel("Accuracy in predicting Block or Distributed RAM selection [%]")
    plt.title("Comparison of HLS, previous FINN method and Decision Tree classifier \n for RAM style predictions (Test set)")
    fig.savefig('../test_set_results/test_set_error.png', bbox_inches='tight')

    import pdb; pdb.set_trace()

def train_ram_class_predictor(df):
    
    #DataType Strip on features
    features = ["mh", "mw", "pe", "simd", "wdt", "idt", "act", "mem_mode"]
    for feature in features:
        df[feature] = df[feature].apply(datatype_strip)
        if feature != 'mem_mode':
            df[feature] = df[feature].astype(int)
    df['LUTRAM'] = df['LUTRAM'].replace('-', -1)
    df["mem_width"] = df.pe*df.simd*df.wdt
    df["mem_height"] = (df.mh*df.mw)/(df.pe*df.simd)

    df['output'] = df['Total_BRAM_18K'].apply(lambda x: '1' if x > 0 else '0')

    #get hls and finn estimate data
    df_hls = df[df.apply(lambda r: r.str.contains('hls', case=False).any(), axis=1)]
    df_finn_estimate = df[df.apply(lambda r: r.str.contains('estimate', case=False).any(), axis=1)]
    #get synthesis data
    df = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]

    df = df.reset_index()
    df_hls = df_hls.reset_index()
    df_finn_estimate = df_finn_estimate.reset_index()

    keep_rows_list = df.index[(df['Total_BRAM_18K'] > 0) | (df['LUTRAM'] > 0)].tolist()

    #remove rows where lutram == 0 and bram == 0
    df = df[(df['Total_BRAM_18K'] > 0) | (df['LUTRAM'] > 0)]
    df_hls = df_hls[df_hls.index.isin(keep_rows_list)]
    df_finn_estimate = df_finn_estimate[df_finn_estimate.index.isin(keep_rows_list)]    
    
    #extract features and target
    features = ["mem_width", "mem_height"]

    X = df.loc[:, features].values
    Y = df.loc[:, 'output'].values

    X_hls = df_hls.loc[:, features].values
    X_finn_estimate = df_finn_estimate.loc[:, features].values
    Y_hls = df_hls.loc[:, 'output'].values
    Y_finn_estimate = df_finn_estimate.loc[:, 'output'].values
 
    #check if the estimate, hls and synth features dfs are identical
    assert (X == X_hls).all(), 'X_hls different from X'
    assert (X == X_finn_estimate).all(), 'X_finn_estimate different from X'

    #train, test split
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size = 0.3, shuffle=True, random_state=2021)

    #same split for hls and estimate, random needs to be set to same int value
    X_train_hls, X_test_hls, Y_train_hls, Y_test_hls = model_selection.train_test_split(X_hls, Y_hls, test_size = 0.3, shuffle=True, random_state=2021)
    X_train_finn_estimate, X_test_finn_estimate, Y_train_finn_estimate, Y_test_finn_estimate = model_selection.train_test_split(X_finn_estimate, Y_finn_estimate, test_size = 0.3, shuffle=True, random_state=2021)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
        .format(logreg.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
        .format(logreg.score(X_test, y_test)))

    clf = DecisionTreeClassifier().fit(X_train, y_train)
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
        .format(clf.score(X_train, y_train)))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
        .format(clf.score(X_test, y_test)))

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
        .format(knn.score(X_train, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
        .format(knn.score(X_test, y_test)))

    svm = SVC()
    svm.fit(X_train, y_train)
    print('Accuracy of SVM classifier on training set: {:.2f}'
        .format(svm.score(X_train, y_train)))
    print('Accuracy of SVM classifier on test set: {:.2f}'
        .format(svm.score(X_test, y_test)))
    
    predictor = clf
    #import pdb; pdb.set_trace()
    plot_relative_error_graph(predictor, X_test, y_test, Y_test_hls, Y_test_finn_estimate)
    
    return predictor

def compute_bram(mh, mw, pe, simd, wdt, mem_mode, ram_style = "auto"):

    mem_width = pe*simd*wdt
    mem_height = (mh*mw)/(pe*simd)
    #import pdb; pdb.set_trace()
    if (mem_mode == "decoupled" and ram_style in ["distributed", "ultra"]) or (mem_mode == "external"):
        return 0
    # assuming SDP mode RAMB18s (see UG573 Table 1-10)
    # assuming decoupled (RTL) memory, which is more efficient than const (HLS)
    if mem_width == 1:
        return math.ceil(mem_height / 16384)
    elif mem_width == 2:
        return math.ceil(mem_height / 8192)
    elif mem_width <= 4:
        return (math.ceil(mem_height / 4096)) * (math.ceil(mem_width / 4))
    elif mem_width <= 9:
        return (math.ceil(mem_height / 2048)) * (math.ceil(mem_width / 9))
    elif mem_width <= 18 or mem_height > 512:
        return (math.ceil(mem_height / 1024)) * (math.ceil(mem_width / 18))
    else:
        return (math.ceil(mem_height / 512)) * (math.ceil(mem_width / 36))

def compute_lutram(mh, mw, pe, simd, wdt):
    
    mem_width = pe*simd*wdt
    mem_height = (mh*mw)/(pe*simd)

    return math.ceil(mem_height/64)*mem_width

def estimate_ram(df, predictor):
    #1 - BRAM, 0 - LUTRAM

    #DataType Strip on features
    features = ["mh", "mw", "pe", "simd", "wdt", "idt", "act", "mem_mode"]
    for feature in features:
        df[feature] = df[feature].apply(datatype_strip)
        if feature != 'mem_mode':
            df[feature] = df[feature].astype(int)

    #get hls and finn estimate data
    df_hls = df[df.apply(lambda r: r.str.contains('hls', case=False).any(), axis=1)]
    df_finn_estimate = df[df.apply(lambda r: r.str.contains('estimate', case=False).any(), axis=1)]
    #get synth data
    df_synth = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]

    df_hls = df_hls.reset_index(drop=True)
    df_finn_estimate = df_finn_estimate.reset_index(drop=True)
    df_synth = df_synth.reset_index(drop=True)

    #add to df_finn_estimate
    df_finn_estimate["mem_width"] = df_finn_estimate.pe*df_finn_estimate.simd*df_finn_estimate.wdt
    df_finn_estimate["mem_height"] = (df_finn_estimate.mh*df_finn_estimate.mw)/(df_finn_estimate.pe*df_finn_estimate.simd)

    df_finn_estimate["ram_class"] = df_finn_estimate.apply(lambda x: int(predictor.predict([[x.mem_width, x.mem_height]])), axis=1)
    #predictor.predict([[df_finn_estimate.mem_width, df_finn_estimate.mem_height]])

    #df_finn_estimate["BRAM_new"] = df_finn_estimate['ram_class'].apply(lambda x: (compute_bram(df_finn_estimate.mh, df_finn_estimate.mw, df_finn_estimate.pe, df_finn_estimate.simd, df_finn_estimate.wdt, df_finn_estimate.mem_mode) if x == 1 else 0), axis=1)
    #df_finn_estimate["LUTRAM_new"] = df_finn_estimate['ram_class'].apply(lambda x: (compute_lutram(df_finn_estimate.mh, df_finn_estimate.mw, df_finn_estimate.pe, df_finn_estimate.simd, df_finn_estimate.wdt) if x == 0 else 0), axis=1)

    df_finn_estimate["BRAM_new"] = -1
    df_finn_estimate["LUTRAM_new"] = -1

    df_finn_estimate["BRAM_new"] = df_finn_estimate.apply(lambda x: compute_bram(x.mh, x.mw, x.pe, x.simd, x.wdt, x.mem_mode) if x.ram_class == 1 else 0, axis=1)
    df_finn_estimate["LUTRAM_new"] = df_finn_estimate.apply(lambda x: compute_lutram(x.mh, x.mw, x.pe, x.simd, x.wdt) if x.ram_class == 0 else 0, axis=1)

    return df_finn_estimate, df_synth

predictor = train_ram_class_predictor(df)
df_finn_estimate, df_synth = estimate_ram(df_initial, predictor)

df_synth["Total_BRAM_18K_denom"] = df_synth["Total_BRAM_18K"].apply(lambda x: 1 if x == 0 else x)

df_finn_estimate['synth_result'] = df_synth["Total_BRAM_18K"]
df_finn_estimate['synth_result_denom'] = df_synth["Total_BRAM_18K_denom"]

df_finn_estimate["bram_rel_error_prev"] = df_finn_estimate.apply(lambda x: (abs(x.Total_BRAM_18K - x.synth_result)/x.synth_result_denom)*100, axis=1)
df_finn_estimate["bram_rel_error_new"] = df_finn_estimate.apply(lambda x: (abs(x.BRAM_new - x.synth_result)/x.synth_result_denom)*100, axis=1)

df_plot = pd.DataFrame()
#df_plot["old_finn_est"] = df_finn_estimate["bram_rel_error_prev"]
df_plot["new_finn_est"] = df_finn_estimate["bram_rel_error_new"]

fig = plt.figure(figsize=(20, 11))
boxplot = df_plot.boxplot(showmeans=True, showfliers=False, return_type='dict', color=dict(boxes='black', whiskers='black', medians='r', caps='black'), patch_artist=True)

fig.savefig('../test_set_results/blabla_new.png', bbox_inches='tight')

filepath = "../test_set_results/updated_fclayer_database_finn_estimate.csv"

df_finn_estimate.to_csv(filepath, index = False, header=True)

#import pdb; pdb.set_trace()