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
from generate_regression_model import *
#import pandas.DataFrame.boxplot

#define the worksheet name from finn-resource-dashboard
worksheet_fc_name = "FCLayer_resources"
worksheet_thresh_name = "Thresholding_layer_resources"
worksheet_swu_name = "Sliding_Window_layer_resources"
#define the directory name where to save the graphs
#directory_name = "FCLayer"
#directory_name = "Thresholding"
#directory_name = "Sliding_Window_Unit"
directory_name = "Special"

#define model features and target
#fclayer
features_fc = ["mh", "mw", "pe", "simd", "wdt", "idt", "mem_mode"]
#features = ["mh", "mw", "pe", "simd", "wdt", "idt", "act", "mem_mode"]

#thresholding
features_thresh = ["ich", "pe", "idt", "act", "mem_mode", "ram_style"]
#features = ["ich", "pe", "idt", "act"]
#features = ["ich", "pe", "idt"]

#swu
features_swu = ["ifm_dim", "ifm_ch", "simd", "k", "stride", "idt", "dw", "ram_style"]
#features = ["ifm_dim", "ifm_ch", "simd", "k", "stride", "idt"]
target = "LUT"

##create the directory
new_dir_path = "../graphs/%s" % directory_name
try:
    os.mkdir(new_dir_path)
except OSError:
    print ("Creation of the directory %s failed" % new_dir_path)
else:
    print ("Successfully created the directory %s " % new_dir_path)

#get all records from the selected worksheet
list_of_dicts_fc = get_records_from_resource_dashboard(worksheet_fc_name)
list_of_dicts_thresh = get_records_from_resource_dashboard(worksheet_thresh_name)
list_of_dicts_swu = get_records_from_resource_dashboard(worksheet_swu_name)
# convert list of dicts to dataframe
df_fc = pd.DataFrame(list_of_dicts_fc)
df_thresh = pd.DataFrame(list_of_dicts_thresh)
df_swu = pd.DataFrame(list_of_dicts_swu)
#clean dataframe
df_fc = clean_dataframe(df_fc)
df_thresh = clean_dataframe(df_thresh)
df_swu = clean_dataframe(df_swu)

X_fc, Y_fc, X_fc_hls, Y_fc_hls, X_fc_finn_estimate, Y_fc_finn_estimate, label_encoder_fc = extract_features_and_target(df_fc, features_fc, target)
X_thresh, Y_thresh, X_thresh_hls, Y_thresh_hls, X_thresh_finn_estimate, Y_thresh_finn_estimate, label_encoder_thresh = extract_features_and_target(df_thresh, features_thresh, target)
X_swu, Y_swu, X_swu_hls, Y_swu_hls, X_swu_finn_estimate, Y_swu_finn_estimate, label_encoder_swu = extract_features_and_target(df_swu, features_swu, target)

#Solution to "zero division error" for relative error computation - using abs(x - x_true)/(1 + abs(x_true))
Y_fc_denominator = np.asarray([(abs(x) + 1) if x == 0 else x for x in Y_fc])
Y_relative_error_hls_fc = (abs(Y_fc_hls - Y_fc)/Y_fc_denominator) * 100
Y_relative_error_estimate_fc = (abs(Y_fc_finn_estimate - Y_fc)/Y_fc_denominator) * 100

Y_thresh_denominator = np.asarray([(abs(x) + 1) if x == 0 else x for x in Y_thresh])
Y_relative_error_hls_thresh = (abs(Y_thresh_hls - Y_thresh)/Y_thresh_denominator) * 100
Y_relative_error_estimate_thresh = (abs(Y_thresh_finn_estimate - Y_thresh)/Y_thresh_denominator) * 100

Y_swu_denominator = np.asarray([(abs(x) + 1) if x == 0 else x for x in Y_swu])
Y_relative_error_hls_swu = (abs(Y_swu_hls - Y_swu)/Y_swu_denominator) * 100
Y_relative_error_estimate_swu = (abs(Y_swu_finn_estimate - Y_swu)/Y_swu_denominator) * 100

df_rel_error = pd.DataFrame()
df_rel_error["FCLayer FINN estimates"] = pd.Series(Y_relative_error_estimate_fc)
df_rel_error["FCLayer HLS estimates"] = pd.Series(Y_relative_error_hls_fc)

df_rel_error["Thresholding FINN estimates"] = pd.Series(Y_relative_error_estimate_thresh)
df_rel_error["Thresholding HLS estimates"] = pd.Series(Y_relative_error_hls_thresh)

df_rel_error["SWU FINN estimates"] = pd.Series(Y_relative_error_estimate_swu)
df_rel_error["SWU HLS estimates"] = pd.Series(Y_relative_error_hls_swu)

###bram
target = "Total_BRAM_18K"
X_fc, Y_fc, X_fc_hls, Y_fc_hls, X_fc_finn_estimate, Y_fc_finn_estimate, label_encoder_fc = extract_features_and_target(df_fc, features_fc, target)
X_thresh, Y_thresh, X_thresh_hls, Y_thresh_hls, X_thresh_finn_estimate, Y_thresh_finn_estimate, label_encoder_thresh = extract_features_and_target(df_thresh, features_thresh, target)
X_swu, Y_swu, X_swu_hls, Y_swu_hls, X_swu_finn_estimate, Y_swu_finn_estimate, label_encoder_swu = extract_features_and_target(df_swu, features_swu, target)

#Solution to "zero division error" for relative error computation - using abs(x - x_true)/(1 + abs(x_true))
Y_fc_denominator = np.asarray([(abs(x) + 1) if x == 0 else x for x in Y_fc])
Y_relative_error_hls_fc = (abs(Y_fc_hls - Y_fc)/Y_fc_denominator) * 100
Y_relative_error_estimate_fc = (abs(Y_fc_finn_estimate - Y_fc)/Y_fc_denominator) * 100

Y_thresh_denominator = np.asarray([(abs(x) + 1) if x == 0 else x for x in Y_thresh])
Y_relative_error_hls_thresh = (abs(Y_thresh_hls - Y_thresh)/Y_thresh_denominator) * 100
Y_relative_error_estimate_thresh = (abs(Y_thresh_finn_estimate - Y_thresh)/Y_thresh_denominator) * 100

Y_swu_denominator = np.asarray([(abs(x) + 1) if x == 0 else x for x in Y_swu])
Y_relative_error_hls_swu = (abs(Y_swu_hls - Y_swu)/Y_swu_denominator) * 100
Y_relative_error_estimate_swu = (abs(Y_swu_finn_estimate - Y_swu)/Y_swu_denominator) * 100

df_rel_error_bram = pd.DataFrame()
#df_rel_error_bram["BRAM_18K FINN estimate FCLayer"] = pd.Series(Y_relative_error_estimate_fc)
df_rel_error_bram["BRAM_18K HLS FCLayer"] = pd.Series(Y_relative_error_hls_fc)

df_rel_error_bram["BRAM_18K FINN estimate Thresholding"] = pd.Series(Y_relative_error_estimate_thresh)
df_rel_error_bram["BRAM_18K HLS Thresholding"] = pd.Series(Y_relative_error_hls_thresh)

df_rel_error_bram["BRAM_18K FINN estimate SWU"] = pd.Series(Y_relative_error_estimate_swu)
df_rel_error_bram["BRAM_18K HLS SWU"] = pd.Series(Y_relative_error_hls_swu)

#import pdb; pdb.set_trace()
fig = plt.figure(figsize=(20, 11))
boxplot = df_rel_error.boxplot(showmeans=True, showfliers=False, return_type='dict', color=dict(boxes='black', whiskers='black', medians='r', caps='black'), patch_artist=True)

colors = ['lightgreen', 'lightskyblue', 'lightgreen', 'lightskyblue', 'lightgreen', 'lightskyblue']

for patch, color in zip(boxplot['means'], colors):
    patch.set_markeredgecolor('red')
    patch.set_markerfacecolor('red')
    
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

plt.title("Relative error of HLS and FINN analytical LUT estimates for Fully Connected, Thresholding and Convolutional Layers")
plt.ylabel('Relative Error [%]')
fig.savefig('../graphs/%s/plot_box_luts_rel_error.png' % (directory_name), bbox_inches='tight')

fig = plt.figure(figsize=(20, 11))
boxplot = df_rel_error_bram.boxplot(showmeans=True, showfliers=False, patch_artist=True)

plt.title("Relative error of HLS and FINN analytical BRAM estimates for Fully Connected, Thresholding and Convolutional Layers")
plt.ylabel('Relative Error [%]')
fig.savefig('../graphs/%s/plot_box_bram_rel_error_without_finn_fc.png' % (directory_name), bbox_inches='tight')
