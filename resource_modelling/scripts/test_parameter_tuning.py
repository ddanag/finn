import sklearn 
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from finn.util.gdrive import *
from sklearn.model_selection import GridSearchCV

#define the worksheet name from finn-resource-dashboard
worksheet_name = "FCLayer_resources"
#get all records from the selected worksheet
list_of_dicts = get_records_from_resource_dashboard(worksheet_name)

# convert list of dicts to dataframe
df = pd.DataFrame(list_of_dicts)

#get only synthesis data
df = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]

#get records where mem_mode=external
df = df[df['mem_mode'].astype(str) == 'external']

#get records where act=None
df = df[df['act'].astype(str) == 'None']

#encode wdt, idt
labelencoder = LabelEncoder()
df['wdt_encoded'] = labelencoder.fit_transform(df['wdt'])
df['idt_encoded'] = labelencoder.fit_transform(df['idt'])

#features = ['mh', 'mw', 'pe', 'simd', 'wdt_encoded', 'idt_encoded']
features = ['pe', 'simd']

#extract features
X = df.loc[:, features].values
#print(X)
#extract target
Y = df.loc[:, ['LUT']].values
#print(Y)
#split the data into train/test data sets 30% testing, 70% training
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3, random_state=0)

gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 10, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        },
        cv=5, scoring='explained_variance', verbose=0, n_jobs=-1)

grid_result = gsc.fit(X_train, Y_train.ravel())

print("Best parameters set found on development set:")
print()
print(grid_result.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
print()

best_params = grid_result.best_params_
best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                   coef0=0.1, shrinking=True,
                   tol=0.001, cache_size=200, verbose=False, max_iter=-1)

scoring = {
               'abs_error': 'neg_mean_absolute_error',
               'squared_error': 'neg_mean_squared_error'}

#scores = cross_validate(best_svr, X_train, Y_train, cv=10, scoring=scoring, return_train_score=True)
#return "MAE :", abs(scores['test_abs_error'].mean()), "| RMSE :", math.sqrt(abs(scores['test_squared_error'].mean()))
best_svr=best_svr.fit(X_train, Y_train.ravel())
print(best_svr.score(X_test, Y_test))

Y_test = Y_test.ravel()
Y_predicted = best_svr.predict(X_test)
print(Y_test)
print(Y_predicted)
print(max(Y_test - Y_predicted))
print(min(Y_test - Y_predicted))
#pd.set_option('display.max_columns', 500)
#print(df)
#model = SVR(kernel='rbf', C=1000, gamma='scale', epsilon=0.1)
#print(model.get_params())
