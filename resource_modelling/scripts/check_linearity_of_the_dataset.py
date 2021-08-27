import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as sts
from finn.util.gdrive import *
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
import os
from generate_regression_model import *

#define the worksheet name from finn-resource-dashboard
worksheet_fc_name = "FCLayer_resources"
worksheet_thresh_name = "Thresholding_layer_resources"
worksheet_swu_name = "Sliding_Window_layer_resources"

#define model features and target
#fclayer
features_fc = ["mh", "mw", "pe", "simd", "wdt", "idt", "act", "mem_mode"]
#get all records from the selected worksheet
list_of_dicts_fc = get_records_from_resource_dashboard(worksheet_fc_name)

# convert list of dicts to dataframe
df_fc = pd.DataFrame(list_of_dicts_fc)

#clean dataframe
df_fc = clean_dataframe(df_fc)

X, Y, X_hls, Y_hls, X_finn_estimate, Y_finn_estimate, label_encoder = extract_features_and_target(df_fc, features_fc, 'LUT')

regressor = LinearRegression().fit(X,Y)
print(r2_score(regressor.predict(X), Y))

#import pdb; pdb.set_trace()