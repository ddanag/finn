import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from finn.util.gdrive import *


#define the worksheet name from finn-resource-dashboard
worksheet_name = "FCLayer_resources"

#get all records from the selected worksheet
list_of_dicts = get_records_from_resource_dashboard(worksheet_name)

# convert list of dicts to dataframe
df = pd.DataFrame(list_of_dicts)
print(df)

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

# Read in data
#df = pd.read_csv('fclayer_resources_last_test.csv', encoding='utf-8')
#get only synthesis data
df = df[df.apply(lambda r: r.str.contains('synthesis', case=False).any(), axis=1)]
#get the estimated data
#df_estimate = df[df.apply(lambda r: r.str.contains('estimate', case=False).any(), axis=1)]
# Print Dataframe
print(df)

#encode act, wdt, idt, mem_mode
labelencoder = LabelEncoder()
#df['act_encoded'] = labelencoder.fit_transform(df['act'])
df['wdt_encoded'] = labelencoder.fit_transform(df['wdt'])
df['idt_encoded'] = labelencoder.fit_transform(df['idt'])
df['mem_mode_encoded'] = labelencoder.fit_transform(df['mem_mode'])

pd.set_option('display.max_columns', 500)
print(df)

#features = ['mh', 'mw', 'pe', 'simd', 'act_encoded', 'wdt_encoded', 'idt_encoded', 'mem_mode_encoded']
features = ['mh', 'mw', 'pe', 'simd', 'wdt_encoded', 'idt_encoded', 'mem_mode_encoded']
#features = ['mh', 'mw', 'pe', 'simd']

#extract features
X = df.loc[:, features].values
print(X)
#extract target
Y = df.loc[:, ['LUT']].values
print(Y)
#split the data into train/test data sets 30% testing, 70% training
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.3, random_state=1) #DD check random_state

model = SVR(kernel='rbf', C=1000, gamma='scale', epsilon=0.1) #DD check kernel
svr_model = model.fit(X_train, Y_train.ravel())

Y_predict = svr_model.predict(X_test)

#print (Y_test.ravel())
#print (Y_predict)

fig1 = plt.figure(1)
ax = fig1.gca()

print(X)
print(X_test)
#print(X_test[:, 2])

pe_test_values = X_test[:, 2]

ax.scatter(Y_test.ravel(), pe_test_values, marker="o", s=200, facecolors='none', edgecolors='g', label='test db')
ax.scatter(Y_predict, pe_test_values, marker="^", color="m", s=50, label='predicted')

ax.set_xlabel("LUTs")
ax.set_ylabel("PE")
ax.set_title("PE vs LUTs")
leg = ax.legend()

plt.show()

print(svr_model.score(X_test, Y_test))

#####
#scores_res = model_selection.cross_val_score(svr, X, Y.ravel(), cv=5)

# Print the accuracy of each fold (i.e. 5 as above we asked cv 5)
#print(scores_res)

# And the mean accuracy of all 5 folds.
#print(scores_res.mean())