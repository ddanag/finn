from finn.util.gdrive import *
from generate_regression_model import *

#USER DEFINES
#define the worksheet name from finn-resource-dashboard
worksheet_name = "FCLayer_resources"
#worksheet_name = "Thresholding_layer_resources"
#worksheet_name = "Sliding_Window_layer_resources"
#define a dictionary for dataframe filtering
filtering_dict = {"mem_mode": [ True, "external"], "act": [True, "None"]}
#filtering_dict = {"mem_mode": [ True, "const"], "ram_style": [True, "distributed"]}
#filtering_dict = {"mem_mode": [ True, "external"], "act": [True, "None"]}
#filtering_dict = {"dw": [ True, "1"], "ram_style": [True, "auto"]}
#define a dictionary for selecting the unseen dataset
#filtering_dict_unseen_df = {"ich":[True, "48", "80", "160", "320"], "idt": [True, "DataType.UINT24"]}
filtering_dict_unseen_df = {}
#define model features and target
#features = ["ich", "pe", "idt", "act"]
#features = ["ich", "pe", "idt"]
features = ["mh", "mw", "pe", "simd", "wdt", "idt"]
#features = ["ifm_dim", "ifm_ch", "simd", "k", "stride", "idt"]
target = "Carry"

#define the directory name where to save the graphs
#directory_name = "Thresholding"
directory_name = "FCLayer"
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

#Hint: clean the dataframe after filtering because cleaning 
#takes a while (~5 min for ~22k samples)
df = clean_dataframe(df)

if len(filtering_dict_unseen_df) != 0:
#get the unseen dataframe and remove from df this subset
    df_unseen = filter_dataframe(df, filtering_dict_unseen_df)
    filtering_dict_df = filtering_dict_unseen_df
    for key in filtering_dict_df:
        filtering_dict_df[key][0] = not filtering_dict_df[key][0]  
    df = filter_dataframe(df, filtering_dict_df)

#generate svr estimator and test datasets
#feature_scaler_selection: 0-Normalization; 1-Standardization
svr_estimator, feature_scaler, target_scaler, X_test, Y_test, Y_hls, Y_finn_estimate = generate_regression_model(df, features, target)

#compute metrics on test set
print("Results on test set:")
compute_metrics(svr_estimator, target_scaler, X_test, Y_test, Y_hls, Y_finn_estimate)
#plot relative error graph
plot_relative_error_graph(svr_estimator, target_scaler, X_test, Y_test, Y_hls, Y_finn_estimate, target, directory_name)
#plot pareto graph
plot_pareto_frontier_graph(svr_estimator, target_scaler, X_test, Y_test, Y_hls, Y_finn_estimate, target, directory_name)

if len(filtering_dict_unseen_df) != 0:
    #extract features and target data from unseen dataframe
    X_test_unseen, Y_test_unseen, X_hls_unseen, Y_hls_unseen, X_finn_estimate_unseen, Y_finn_estimate_unseen = extract_features_and_target(df_unseen, features, target)
    #apply scaler on features
    X_test_unseen = apply_feature_scaler(feature_scaler, X_test_unseen)

    #compute metrics on unseen test set
    print("Results on unseen test set:")
    compute_metrics(svr_estimator, target_scaler, X_test_unseen, Y_test_unseen, Y_hls_unseen, Y_finn_estimate_unseen)
