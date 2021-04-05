from finn.util.gdrive import *
from generate_regression_model import *

#USER DEFINES
#define the worksheet name from finn-resource-dashboard
worksheet_name = "Thresholding_layer_resources_old_mixed_with_test"
#define a dictionary for dataframe filtering
filtering_dict = {"mem_mode": [ True, "decoupled"], "ram_style": [True, "distributed"], "act": [True, "DataType.BIPOLAR"]}
#define a dictionary for selecting the unseen dataset
filtering_dict_unseen_df = {"idt": [True, "DataType.UINT14", "DataType.UINT18", "DataType.UINT22",  "DataType.UINT26",  "DataType.UINT30"], "ich":[True, "48", "80", "160", "320"]}
#define model features and target
features = ["ich", "pe", "idt"]
target = "LUT"

#get all records from the selected worksheet
list_of_dicts = get_records_from_resource_dashboard(worksheet_name)
# convert list of dicts to dataframe
df = pd.DataFrame(list_of_dicts)

#filter dataframe
df = filter_dataframe(df, filtering_dict)

#Hint: clean the dataframe after filtering because cleaning 
#takes a while (~5 min for ~22k samples)
df = clean_dataframe(df)

#get the unseen dataframe and remove from df this subset
df_unseen = filter_dataframe(df, filtering_dict_unseen_df)
filtering_dict_df = filtering_dict_unseen_df
for key in filtering_dict_df:
    filtering_dict_df[key][0] = not filtering_dict_df[key][0]  
df = filter_dataframe(df, filtering_dict_df)

#generate svr estimator and test datasets
#scaler_selection: 0-Normalization; 1-Standardization
svr_estimator, scaler, X_test, Y_test, Y_hls, Y_finn_estimate = generate_regression_model(df, features, target)

#compute metrics on test set
print("Results on test set:")
compute_metrics(svr_estimator, X_test, Y_test, Y_hls, Y_finn_estimate)

#extract features and target data from unseen dataframe
X_test_unseen, Y_test_unseen, X_hls_unseen, Y_hls_unseen, X_finn_estimate_unseen, Y_finn_estimate_unseen = extract_features_and_target(df_unseen, features, target)
#apply scaler on features
X_test_unseen = apply_scaler(scaler, X_test_unseen)

#compute metrics on unseen test set
print("Results on unseen test set:")
compute_metrics(svr_estimator, X_test_unseen, Y_test_unseen, Y_hls_unseen, Y_finn_estimate_unseen)
