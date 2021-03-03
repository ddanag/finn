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

#### applies to FCLayer database
#choose activation option 'None' or 'Other'
act_option = 'None'
#split the database based on type of act - None or Other
if "FCLayer" in worksheet_name:
    mask = df['act'].astype(str) == 'None'
    df_act_None = df[mask]
    df_act_Other = df[~mask]

    if act_option == 'None':
        df = df_act_None
        del df_act_None['act']
    else:
        df = df_act_Other
        del df_act_Other['act']
####

#separate the dataframe in 3 sub-dataframes:
#   - one which contains the estimated resources
#   - one which contains the hls estimated resources
#   - one which contains the resources reported after Vivado synthesis
df_estimate = df[df.apply(lambda r: r.str.contains('estimate', case=False).any(), axis=1)]
df_hls = df[df.apply(lambda r: r.str.contains('hls', case=False).any(), axis=1)]
df_synth = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]

def get_SVR_estimates(res_class):

    #encode act, wdt, idt, mem_mode
    labelencoder = LabelEncoder()
    #df_synth['act_encoded'] = labelencoder.fit_transform(df_synth['act'])
    df_synth['wdt_encoded'] = labelencoder.fit_transform(df_synth['wdt'])
    df_synth['idt_encoded'] = labelencoder.fit_transform(df_synth['idt'])
    df_synth['mem_mode_encoded'] = labelencoder.fit_transform(df_synth['mem_mode'])

    #features = ['mh', 'mw', 'pe', 'simd', 'act_encoded', 'wdt_encoded', 'idt_encoded', 'mem_mode_encoded']
    features = ['mh', 'mw', 'pe', 'simd', 'wdt_encoded', 'idt_encoded', 'mem_mode_encoded']

    #extract features
    X_train = df_synth.loc[:, features].values
    #extract target
    Y_train = df_synth.loc[:, [res_class]].values

    #define the SVR model and its hyperparameters
    model = SVR(kernel='rbf', C=10000, gamma='scale', epsilon=0.1) #DD check kernel
    #train the SVR model
    svr_model = model.fit(X_train, Y_train.ravel())

    Y_predict = svr_model.predict(X_train)

    return Y_predict

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
        ax.set_title("%s vs %s (FPGA = %s, act = %s)" % (parameter, res_class, fpga, act_option))
    else:
        ax.set_title("%s vs %s (FPGA = %s)" % (parameter, res_class, fpga))
    
    leg = ax.legend()
    
    fig.savefig('../graphs/%s/plot_%s_vs_%s.png' % (directory_name, parameter, res_class), bbox_inches='tight')

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
#parameters = ['pe', 'simd']
#res_classes = ['LUT']

for parameter in parameters:
    for res_class in res_classes:
        generate_graph (parameter, res_class)
    generate_graph(parameter, 'Delay')

