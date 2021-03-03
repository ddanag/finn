import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as sts
from finn.util.gdrive import *
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import os

#define the worksheet name from finn-resource-dashboard
worksheet_name = "FCLayer_resources"
#define the directory name where to save the graphs
directory_name = "FCLayer"

##create the directory
new_dir_path = "../graphs/%s" % directory_name
try:
    os.mkdir(new_dir_path)
except OSError:
    print ("Creation of the directory %s failed" % new_dir_path)
else:
    print ("Successfully created the directory %s " % new_dir_path)

#get all records from the selected worksheet
list_of_dicts = get_records_from_resource_dashboard(worksheet_name)

# convert list of dicts to dataframe
df = pd.DataFrame(list_of_dicts)
print(df)

fpga = df['FPGA'].iloc[0]

#get records where mem_mode=external
df = df[df['mem_mode'].astype(str) == 'external']

#get records where act=None
df = df[df['act'].astype(str) == 'None']

#separate the dataframe in 3 sub-dataframes:
#   - one which contains the estimated resources
#   - one which contains the hls estimated resources
#   - one which contains the resources reported after Vivado synthesis
df_estimate = df[df.apply(lambda r: r.str.contains('estimate', case=False).any(), axis=1)]
df_hls = df[df.apply(lambda r: r.str.contains('hls', case=False).any(), axis=1)]
df_synth = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]

def models(res_class):

    #encode wdt, idt
    labelencoder = LabelEncoder()
    df_synth['wdt_encoded'] = labelencoder.fit_transform(df_synth['wdt'])
    df_synth['idt_encoded'] = labelencoder.fit_transform(df_synth['idt'])

    features = ['mh', 'mw', 'pe', 'simd', 'wdt_encoded', 'idt_encoded']
    #features = ['mh', 'mw', 'pe', 'simd']
    #extract features
    X = df_synth.loc[:, features].values
    #extract target
    Y = df_synth.loc[:, [res_class]].values

    #split the data into train/test data sets 30% testing, 70% training
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3, random_state=0)
    
    #linear regression
    linear_reg_model = LinearRegression()
    linear_reg_model = linear_reg_model.fit(X_train, Y_train)
    Y_predict_linear = linear_reg_model.predict(X_test)
    score_linear = linear_reg_model.score(X_test, Y_test)

    #search for the best SVR hyperparameters
    gscv = GridSearchCV(
        estimator=SVR(kernel='poly', max_iter=20000000),
        param_grid={
            #'C': [0.1, 1, 10, 100, 1000],
            #'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            #'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
            'C': [100],
            'epsilon': [5],
            'gamma': [0.005]
        },
        cv=5, n_jobs = -1, verbose = 2)

    grid_result = gscv.fit(X_train, Y_train.ravel())

    print("Best parameters set found on development set:")
    print()
    print(grid_result.best_params_)

    #get best hyperparameters and define the model
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='poly', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"])
    
    #train the SVR model
    best_svr = best_svr.fit(X_train, Y_train.ravel())

    Y_predict_svr = best_svr.predict(X_test)

    print(X_test)

    print(best_svr.score(X_test, Y_test))

    #cross-validation
    #scores = cross_val_score(best_svr, X, Y.ravel(), cv=5)
    #print(scores)

    print(best_svr.get_params(deep=True))
    print(len(best_svr.get_params(deep=True)))
    print(best_svr.support_vectors_)

    return X_test, Y_test, Y_predict_svr, best_svr.score(X_test, Y_test), Y_predict_linear, score_linear

def generate_graph_test_set(parameter, res_class):

    X_test, Y_test, Y_predict_svr, score_svr, Y_predict_linear, score_linear = models(res_class)

    fig = plt.figure(figsize=(20, 11))
    ax = fig.gca()

    X_test = [col[2] for col in X_test]

    ax.scatter(Y_predict_linear, X_test, marker="o", s=200, facecolors='none', edgecolors='g', label='predicted_linear')
    ax.scatter(Y_predict_svr, X_test, marker="^", s=500, facecolors='none', edgecolors='m', label='predicted_svr')
    ax.scatter(Y_test, X_test, marker="x", s=50, color='r', label='synth')

    ax.set_xlabel("%s" % 'LUT')
    ax.set_ylabel("%s" % 'pe')

    if "FCLayer" in worksheet_name:
        ax.set_title("%s vs %s (SVR_score = %s, linear_score = %s)" % ('pe', 'LUT', score_svr, score_linear))
    else:
        ax.set_title("%s vs %s (SVR_score = %s, linear_score = %s)" % ('pe', 'LUT', score_svr, score_linear))
    
    leg = ax.legend()
    
    fig.savefig('../graphs/%s/plot_%s_vs_%s_MAC_test.png' % (directory_name, 'pe', 'LUT'), bbox_inches='tight')


#get dataframe headers
headers = list(df)
print('All headers:', headers)

#get all parameters and resource classes from the csv file
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
parameters.remove('FPGA')
res_classes.remove('finn_commit')
res_classes.remove('vivado_version')
res_classes.remove('vivado_build_no')

#separate resource classes and timing info
timing_info = ['TargetClockPeriod', 'EstimatedClockPeriod', 'Delay', 'TargetClockFrequency [MHz]', 'EstimatedClockFrequency [MHz]']
res_classes = [i for i in res_classes if i not in timing_info]

print('Parameters list:', parameters)
print('Resource classes:', res_classes)
print('Timing info:', timing_info)

#for testing
parameters = ['pe']
res_classes = ['LUT']

for parameter in parameters:
    for res_class in res_classes:
        #generate_graph (parameter, res_class)
        #predicted_res = get_SVR_estimates(res_class)
        generate_graph_test_set(parameter, res_class)
    #generate_graph(parameter, 'Delay')

"""
def get_SVR_estimates(res_class):

    #encode wdt, idt
    labelencoder = LabelEncoder()
    df_synth['wdt_encoded'] = labelencoder.fit_transform(df_synth['wdt'])
    df_synth['idt_encoded'] = labelencoder.fit_transform(df_synth['idt'])

    features = ['mh', 'mw', 'pe', 'simd', 'wdt_encoded', 'idt_encoded']
    #features = ['pe', 'simd',]
    #extract features
    X = df_synth.loc[:, features].values
    #extract target
    Y = df_synth.loc[:, [res_class]].values

    #split the data into train/test data sets 30% testing, 70% training
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3, random_state=0)
    
    #search for the best SVR hyperparameters
    gscv = GridSearchCV(
        estimator=SVR(kernel='poly'),
        param_grid={
            #'C': [0.1, 1, 10, 100, 1000],
            #'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            #'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
            'C': [100],
            'epsilon': [0.01],
            'gamma': [0.1]
        },
        cv=5, n_jobs = -1, verbose = 2)

    grid_result = gscv.fit(X_train, Y_train.ravel())

    print("Best parameters set found on development set:")
    print()
    print(grid_result.best_params_)

    #get best hyperparameters and define the model
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='poly', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"])
    
    #train the SVR model
    best_svr = best_svr.fit(X_train, Y_train.ravel())

    Y_predict_whole_db = best_svr.predict(X)
    Y_predict_test_db = best_svr.predict(X_test)

    print(X_test)

    print(best_svr.score(X_test, Y_test))

    generate_graph_test_set(X_test, Y_test, Y_predict_test_db, best_svr.score(X_test, Y_test))

    return Y_predict_whole_db

def generate_graph_test_set(x_test, y_test, y_predicted, score):
    fig = plt.figure(figsize=(20, 11))
    ax = fig.gca()

    x_test = [col[2] for col in x_test]

    ax.scatter(y_predicted, x_test, marker="^", s=500, facecolors='none', edgecolors='m', label='predicted')
    ax.scatter(y_test, x_test, marker="x", s=50, edgecolors='r', label='synth')

    ax.set_xlabel("%s" % 'LUT')
    ax.set_ylabel("%s" % 'pe')

    if "FCLayer" in worksheet_name:
        ax.set_title("%s vs %s (SVR_score = %s)" % ('pe', 'LUT', score))
    else:
        ax.set_title("%s vs %s (SVR_score = %s)" % ('pe', 'LUT', score))
    
    leg = ax.legend()
    
    fig.savefig('../graphs/%s/plot_%s_vs_%s_MAC_test.png' % (directory_name, 'pe', 'LUT'), bbox_inches='tight')


def generate_graph (parameter, res_class):
    fig = plt.figure(figsize=(20, 11))
    ax = fig.gca()

    #plot predicted resources
    predicted_res = get_SVR_estimates(res_class)
    ax.scatter(predicted_res, df_synth[parameter], marker="^", s=500, facecolors='none', edgecolors='m', label='predicted')

    #plot estimated, hls, synth resources
    ax.scatter(df_estimate[res_class], df_estimate[parameter], marker="o", s=200, facecolors='none', edgecolors='g', label='estimate')
    ax.scatter(df_hls[res_class], df_hls[parameter], marker="s", s=100, facecolors='none', edgecolors='b', label='hls')
    ax.scatter(df_synth[res_class], df_synth[parameter], marker="x", s=50, color="r", label='synth')
    
    #compute and plot mean, median and standard deviation
    mean_est = df_estimate[res_class].mean()
    median_est = df_estimate[res_class].median()
    std_est = df_estimate[res_class].std()

    mean_hls = df_hls[res_class].mean()
    median_hls = df_hls[res_class].median()
    std_hls = df_hls[res_class].std()

    mean_synth = df_synth[res_class].mean()
    median_synth = df_synth[res_class].median()
    std_synth = df_synth[res_class].std()

    mean_predicted = predicted_res.mean()
    median_predicted = np.median(predicted_res)
    std_predicted = predicted_res.std()

    ax.axvline(mean_est, color='g', linestyle='--', label='mean_est')
    ax.axvline(median_est, color='g', linestyle='-', label='median_est')

    ax.axvline(mean_hls, color='b', linestyle='--', label='mean_hls')
    ax.axvline(median_hls, color='b', linestyle='-', label='median_hls')

    ax.axvline(mean_synth, color='r', linestyle='--', label='mean_synth')
    ax.axvline(median_synth, color='r', linestyle='-', label='median_synth')
    
    ax.axvline(mean_predicted, color='m', linestyle='--', label='mean_predicted')
    ax.axvline(median_predicted, color='m', linestyle='-', label='median_predicted')

    #compute Spearman and Pearson correlation coefficients
    #Spearman:   +1 - monotonically increasing relationship
    #            -1 - monotonically decreasing relationship
    #Pearson:   +1 - total positive linear correlation
    #           -1 - total negative linear correlation
    min_len = min(len(df_estimate[res_class]), len(df_hls[res_class]))
    spearman_corr_est_hls = sts.spearmanr(df_estimate[res_class][:min_len], df_hls[res_class][:min_len])
    pearson_corr_est_hls = sts.pearsonr(df_estimate[res_class][:min_len], df_hls[res_class][:min_len])

    min_len = min(len(df_estimate[res_class]), len(df_synth[res_class]))
    spearman_corr_est_synth = sts.spearmanr(df_estimate[res_class][:min_len], df_synth[res_class][:min_len])
    pearson_corr_est_synth = sts.pearsonr(df_estimate[res_class][:min_len], df_hls[res_class][:min_len])

    #some params are str, can't compute corr
    #DD check - use encoding to compute corr?
    try:
        spearman_corr_param_synth_resource = sts.spearmanr(df_synth[parameter], df_synth[res_class])
        pearson_corr_param_synth_resource = sts.pearsonr(df_synth[parameter], df_synth[res_class])
        textstr = '\n'.join((
        'mean_est=%.2f' % (mean_est),
        'median_est=%.2f' % (median_est),
        'std_est=%.2f' % (std_est),
        ' ',
        'mean_hls=%.2f' % (mean_hls),
        'median_hls=%.2f' % (median_hls),
        'std_hls=%.2f' % (std_hls),
        ' ',
        'mean_synth=%.2f' % (mean_synth),
        'median_synth=%.2f' % (median_synth),
        'std_synth=%.2f' % (std_synth),
        ' ',
        'Correlation est - hls',
        'spearman_corr=%.2f' % (spearman_corr_est_hls.correlation),
        'pearson_corr=%.2f' % (pearson_corr_est_hls[0]),
        ' ',
        'Correlation est - synth',
        'spearman_corr=%.2f' % (spearman_corr_est_synth.correlation),
        'pearson_corr=%.2f' % (pearson_corr_est_synth[0]),
        ' ',
        'Correlation %s - synth %s' % (parameter, res_class), 
        'spearman_corr=%.2f' % (spearman_corr_param_synth_resource.correlation),
        'pearson_corr=%.2f' % (pearson_corr_param_synth_resource[0]),
        ' ',
        ))
    except:
        spearman_corr_param_synth_resource = None
        pearson_corr_param_synth_resource = None
        textstr = '\n'.join((
        'mean_est=%.2f' % (mean_est),
        'median_est=%.2f' % (median_est),
        'std_est=%.2f' % (std_est),
        ' ',
        'mean_hls=%.2f' % (mean_hls),
        'median_hls=%.2f' % (median_hls),
        'std_hls=%.2f' % (std_hls),
        ' ',
        'mean_synth=%.2f' % (mean_synth),
        'median_synth=%.2f' % (median_synth),
        'std_synth=%.2f' % (std_synth),
        ' ',
        'Correlation est - hls',
        'spearman_corr=%.2f' % (spearman_corr_est_hls.correlation),
        'pearson_corr=%.2f' % (pearson_corr_est_hls[0]),
        ' ',
        'Correlation est - synth',
        'spearman_corr=%.2f' % (spearman_corr_est_synth.correlation),
        'pearson_corr=%.2f' % (pearson_corr_est_synth[0]),
        ' ',
        ))

    plt.subplots_adjust(right=0.8)
    # figtext() takes positional arguments x (0.93) and y (0.5) and a string. The bbox=dict(facecolor='white') creates a box around the text with a white facecolor.
    side_text = plt.figtext(0.82, 0.5, textstr, bbox=dict(facecolor='white'))

    ax.set_xlabel("%s" % res_class)
    ax.set_ylabel("%s" % parameter)

    if "FCLayer" in worksheet_name:
        ax.set_title("%s vs %s (FPGA = %s)" % (parameter, res_class, fpga))
    else:
        ax.set_title("%s vs %s (FPGA = %s)" % (parameter, res_class, fpga))
    
    leg = ax.legend()
    
    fig.savefig('../graphs/%s/plot_%s_vs_%s_MAC.png' % (directory_name, parameter, res_class), bbox_inches='tight')
"""