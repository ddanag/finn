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
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import os, csv

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

filename  = "db_act.csv"

#get all records from the selected worksheet
list_of_dicts = get_records_from_resource_dashboard(worksheet_name)

# convert list of dicts to dataframe
df = pd.DataFrame(list_of_dicts)
print(df)

pd.set_option('display.max_columns', 500)

fpga = df['FPGA'].iloc[0]

#get synth data
df = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]

#get records where mem_mode=external
df = df[df['mem_mode'].astype(str) == 'external']

#get records where act!=None
df_act = df[df['act'].astype(str) != 'None']
print('here1')
print(df_act)
df_act = df_act[df_act['act'].astype(str) != 'DataType.BIPOLAR']
print('here2')
print(df_act)
#get records where act=None
df_act_none = df[df['act'].astype(str) == 'None']

def models(res_class):

    #encode wdt, idt
    labelencoder = LabelEncoder()
    df_training['wdt_encoded'] = labelencoder.fit_transform(df_training['wdt'])
    df_training['idt_encoded'] = labelencoder.fit_transform(df_training['idt'])

    features = ['mh', 'mw', 'pe', 'simd', 'wdt_encoded', 'idt_encoded']
    #features = ['mh', 'mw', 'pe', 'simd']
    #extract features
    X = df_training.loc[:, features].values
    #extract target
    Y = df_training.loc[:, [res_class]].values

    #split the data into train/test data sets 30% testing, 70% training
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3, random_state=0)
    
    #linear regression
    linear_reg_model = LinearRegression()
    linear_reg_model = linear_reg_model.fit(X_train, Y_train)
    Y_predict_linear = linear_reg_model.predict(X_test)
    score_linear = linear_reg_model.score(X_test, Y_test)

    ####
    #model_pipeline = Pipeline([('model', TransformedTargetRegressor(regressor = SVR(max_iter=20000000), transformer=MinMaxScaler()))])
    model_pipeline = TransformedTargetRegressor(regressor = SVR(max_iter = 20000000), transformer=MinMaxScaler())
    #search for the best SVR hyperparameters
    gscv = GridSearchCV(
        estimator=model_pipeline,
        param_grid={
            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            #'model__regressor__C': [0.1, 1, 10, 100, 1000],
            #'model__regressor__epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            #'model__regressor__gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
            'regressor__kernel': ['poly'],
            'regressor__C': [1],
            'regressor__epsilon': [1],
            'regressor__gamma': [0.01]
        },
        cv=5, n_jobs = -1, verbose = 2)

    grid_result = gscv.fit(X_train, Y_train.ravel())

    print("Best parameters set found on development set:")
    print()
    print(grid_result.best_params_)

    #get best hyperparameters and define the model
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='poly', C=best_params["regressor__C"], epsilon=best_params["regressor__epsilon"], gamma=best_params["regressor__gamma"])
    #best_svr = TransformedTargetRegressor(regressor = SVR(kernel=best_params["regressor__kernel"], C=best_params["regressor__C"], epsilon=best_params["regressor__epsilon"], gamma=best_params["regressor__gamma"]), transformer=MinMaxScaler())
    #train the SVR model
    best_svr = best_svr.fit(X_train, Y_train.ravel())

    Y_predict_svr = best_svr.predict(X_test)
    #Y_predict_svr = grid_result.predict(X_test)
    print(X_test)

    print(best_svr.score(X_test, Y_test))
    #print('grid score')
    #print(grid_result.score(X_test, Y_test))

    #cross-validation
    svr_scores = cross_val_score(best_svr, X, Y.ravel(), scoring='neg_mean_absolute_error', cv=10)
    svr_scores = np.absolute(svr_scores)
    print('Mean MAE SVR:', np.mean(svr_scores))

    lin_scores = cross_val_score(linear_reg_model, X, Y.ravel(), scoring='neg_mean_absolute_error', cv=10)
    lin_scores = np.absolute(lin_scores)
    print('Mean MAE linear:', np.mean(lin_scores))

    #return X_test, Y_test, Y_predict_svr, grid_result.score(X_test, Y_test), Y_predict_linear, score_linear
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
    
    fig.savefig('../graphs/%s/plot_%s_vs_%s_act_target_scaling_test.png' % (directory_name, 'pe', 'LUT'), bbox_inches='tight')


#get dataframe headers
headers = list(df)

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
#remove timing 
#remove act, mem_mode, Res from
columns_to_remove = ['act', 'mem_mode', 'Resources from:', 'FPGA', 'finn_commit', 'vivado_version', 'vivado_build_no', 'TargetClockPeriod', 'EstimatedClockPeriod', 'Delay', 'TargetClockFrequency [MHz]', 'EstimatedClockFrequency [MHz]']
res_classes = [element for element in res_classes if element not in columns_to_remove]
parameters = [element for element in parameters if element not in columns_to_remove]

df_act = df_act.drop(columns_to_remove, axis=1)
df_act_none = df_act_none.drop(columns_to_remove, axis=1)

print(parameters)
print(res_classes)
print(len(df_act))
print(len(df_act_none))
print(df_act)
print(df_act_none)



#isolate contribution of weights to overall resource utilization by subtracting resources of equivalent (mem_mode = external) configuration 
df_training = pd.DataFrame(columns=list(df_act))

with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(list(df_training))

found_row = False
for index, row1 in df_act_none.iterrows():
    for index, row2 in df_act.iterrows():
        for s in parameters:
            if row1[s] == row2[s]:
                found_row = True
            else:
                found_row = False
                break
        if found_row == True:
            print(row1)
            print(row2)
            for res in res_classes:
                row2[res] = row2[res] - row1[res]
            df_training = df_training.append(row2, ignore_index=True)

            with open(filename, 'a+') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(row2)

print(df_training)

#for testing
parameters = ['pe']
res_classes = ['LUT']

for parameter in parameters:
    for res_class in res_classes:
        generate_graph_test_set(parameter, res_class)
        