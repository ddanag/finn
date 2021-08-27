from finn.util.gdrive import *
from generate_regression_model import *
import json
import numpy as np

#for testing restoring the model from json file
from sklearn.svm import SVR

#USER DEFINES
#define the worksheet name from finn-resource-dashboard
worksheet_name = "FCLayer_resources"
#worksheet_name = "Thresholding_layer_resources"
#worksheet_name = "Sliding_Window_layer_resources"

#define a dictionary for dataframe filtering
#filtering_dict = {"act": [True, "None"], "mem_mode": [ True, "decoupled"]}
#filtering_dict = {"act": [True, "None"]}
#filtering_dict = {"mem_mode": [ True, "const"], "ram_style": [True, "distributed"]}
#filtering_dict = {"dw": [ True, "0"]}
#filtering_dict = {"dw": [ True, "0"], "ram_style": [True, "distributed"]}
#filtering_dict = {"idt": [ False, "DataType.BIPOLAR"], "wdt": [False, "DataType.BIPOLAR"]}
#filtering_dict = {"mem_mode": [ True, "decoupled"]}
filtering_dict = {}

#define a dictionary for selecting the unseen dataset
#filtering_dict_unseen_df = {"ich":[True, "48", "80", "160", "320"], "idt": [True, "DataType.UINT24"]}
#filtering_dict_unseen_df = {"mh":[True, "512", "1024"], "mw":[True, "512", "1024", "2048"]}
#filtering_dict_unseen_df = {"mh":[True, "1024"]}
#filtering_dict_unseen_df = {"mh":[True, "1024"], "idt": [True, "DataType.INT4"], "wdt": [True, "DataType.INT4"]}
filtering_dict_unseen_df = {}

#define model features and target
#fclayer
#features = ["mh", "mw", "pe", "simd", "wdt", "idt"]
#features = ["mh", "mw", "pe", "simd"]
#features = ["mh", "mw", "pe", "simd", "wdt", "idt", "mem_mode"]
features = ["mh", "mw", "pe", "simd", "wdt", "idt", "act", "mem_mode"]

#thresholding
#features = ["ich", "pe", "idt", "act", "mem_mode", "ram_style"]
#features = ["ich", "pe", "idt", "act", "mem_mode"]
#features = ["ich", "pe", "idt", "act"]
#features = ["ich", "pe", "idt"]

#swu
#features = ["ifm_dim", "ifm_ch", "simd", "k", "stride", "idt", "dw", "ram_style"]
#features = ["ifm_dim", "ifm_ch", "simd", "k", "stride", "idt", "ram_style"]
#features = ["ifm_dim", "ifm_ch", "simd", "k", "stride", "idt", "dw"]
#features = ["ifm_dim", "ifm_ch", "simd", "k", "stride", "idt"]
target = "LUT"

#target_scaler:   0 - log
#                 1 - (synth-finn_estimate)
#                 None  
target_scaler = None

#define the directory name where to save the graphs
directory_name = "FCLayer"
#directory_name = "Thresholding"
#directory_name = "Sliding_Window_Unit"
#create the directory where to save the graphs
new_dir_path = "../graphs/%s" % directory_name
try:
    os.mkdir(new_dir_path)
except OSError:
    print ("Creation of the directory %s failed" % new_dir_path)
else:
    print ("Successfully created the directory %s" % new_dir_path)

#get all records from the selected worksheet
list_of_dicts = get_records_from_resource_dashboard(worksheet_name)
# convert list of dicts to dataframe
df = pd.DataFrame(list_of_dicts)

#filter dataframe
df = filter_dataframe(df, filtering_dict)

#remove the fully unfolded configurations - outliers
#df = remove_fully_unfolded_configs(df, directory_name)

#Hint: clean the dataframe after filtering because cleaning 
#takes a while (~5 min for ~22k samples)
df = clean_dataframe(df)
#import pdb; pdb.set_trace()
if len(filtering_dict_unseen_df) != 0:
#get the unseen dataframe and remove from df this subset
    df_unseen = filter_dataframe(df, filtering_dict_unseen_df)
    filtering_dict_df = filtering_dict_unseen_df
    for key in filtering_dict_df:
        filtering_dict_df[key][0] = not filtering_dict_df[key][0]  
    df = filter_dataframe(df, filtering_dict_df)

#generate svr estimator and test datasets
#feature_scaler_selection: 0-Normalization; 1-Standardization
svr_estimator, feature_scaler, target_scaler, label_encoder, X_train_before, X_train, Y_train, X_test, Y_test, Y_hls, Y_finn_estimate, X_test_before_processing = generate_regression_model(df, features, target, target_scaler = target_scaler)

#the method for saving and restoring the model is the worst possible.
#TODO need to fix this ASAP

#save the svr_estimator
attributes = {}
estimator_params = svr_estimator.get_params()

#save the attributes
"""
attributes['class_weight_'] = (svr_estimator.class_weight_).tolist()
if svr_estimator.kernel == 'linear':
    attributes['coef_'] = (svr_estimator.coef_).tolist()
   
attributes['dual_coef_'] = (svr_estimator.dual_coef_).tolist()
attributes['fit_status_'] = svr_estimator.fit_status_
attributes['intercept_'] = (svr_estimator.intercept_).tolist()
attributes['_n_support'] = (svr_estimator._n_support).tolist()
attributes['shape_fit_'] = svr_estimator.shape_fit_
attributes['support_'] = (svr_estimator.support_).tolist()
attributes['support_vectors_'] = (svr_estimator.support_vectors_).tolist()
"""

dict_to_write = {}
dict_to_write['params'] = estimator_params
dict_to_write['X_train_before'] = X_train_before.tolist()
dict_to_write['X_train'] = X_train.tolist()
dict_to_write['Y_train'] = Y_train.tolist()
try:
    dict_to_write['label_classes'] = label_encoder.classes_.tolist()
except:
    print("There are no label encoder classes.")
dict_to_write['target_scaler'] = target_scaler

with open('../models/%s_%s_model.json' %(directory_name,target), 'w') as file:
    json.dump(dict_to_write, file)

#compute metrics on test set
print("Results on test set:")
compute_metrics(svr_estimator, target_scaler, X_test, Y_test, Y_hls, Y_finn_estimate)
#plot relative error graph
plot_relative_error_graph(svr_estimator, target_scaler, X_test, Y_test, Y_hls, Y_finn_estimate, target, directory_name)
#plot pareto graph
plot_pareto_frontier_graph(svr_estimator, target_scaler, X_test, Y_test, Y_hls, Y_finn_estimate, target, directory_name)

"""
if len(filtering_dict_unseen_df) != 0:
    #extract features and target data from unseen dataframe
    X_test_unseen, Y_test_unseen, X_hls_unseen, Y_hls_unseen, X_finn_estimate_unseen, Y_finn_estimate_unseen = extract_features_and_target(df_unseen, features, target)
    #apply scaler on features
    X_test_unseen = apply_feature_scaler(feature_scaler, X_test_unseen)

    #compute metrics on unseen test set
    print("Results on unseen test set:")
    compute_metrics(svr_estimator, target_scaler, X_test_unseen, Y_test_unseen, Y_hls_unseen, Y_finn_estimate_unseen)
"""
save_test_results_to_csv(svr_estimator, target_scaler, label_encoder, X_test_before_processing, X_test, Y_test, Y_hls, Y_finn_estimate, target, directory_name, features)