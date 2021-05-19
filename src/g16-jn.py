# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Supervised Learning
# ## Disease Classification based on symptoms 
# ### (Covid-19, Flu, Cold, Allergy)
# %% [markdown]
# Professor: 
# * Luís Paulo Reis
# 
# Group members:
# * Eduardo Brito, up201806271
# * Paulo Ribeiro, up201806505
# * Rita Silva, up201806527
# 
# ### Specification
# 
# **Covid-19**, the common **Cold**, **Seasonal Allergies** and the **Flu** have many similar signs and
# symptoms. These common problems are often mistaken for Covid-19 and this project will help
# provide a distinction between them.
# 
# Based on a data set with information about some patients’ diagnosis and the experienced
# symptoms: 
# 
# > *Our goal is to associate them and understand their relationship in order to help diagnose
# new patients.*
# 
# Hereupon, we identify this as **a single label multiclass classification problem** with 21 attributes:
# 
# - 20  distinct  symptoms,  with  a  value  of  1  if  the  patient  suffers  from  it  and  0  otherwise
# - 1 diagnose with 4 possible outcomes (Covid-19, Cold, Allergies and Flu).
# 
# ### Tools & Resources
# 
# * Data Preprocessing:
#     * pandas
#     * numpy
# * Data Visualization:
#     * seaborn
#     * matplotlib
# * ML Algorithms:
#     * scikitlearn
# 
# ### Data Analysis & Preprocessing

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os.path


# %%
# Load csv data
data = pd.read_csv("../data/large_data.csv")


# %%
# Print head values and summary statistics
print(data.head())
print()
print(data.describe())
print()


# %%
# Get separate list of symptoms and diseases
columns = data.columns
symptoms = list(columns)
print("Symptoms: {}".format(symptoms))
print()
symptoms.remove("TYPE")
diseases = list(data["TYPE"].unique())
print("Diseases: {}".format(diseases))


# %%
# Check for columns with not allowed values
print("Checking for columns with not allowed values:")
found = False
for i in symptoms:
    values = data[i].unique()
    # Values different from 0 and 1
    if not np.array_equiv(np.sort(values), np.array([0,1])):
        found = True
        print("Column {} has different values: {}".format(i, values))

if not found: print("Nothing Found...")

# %% [markdown]
# **Based on the summary, our problem presents some properties:**
# * Nominal and Discrete binary attributes
# * Dimensionality = 21 attributes
# * Size = 44k records
# * Type = Data Matrix
# * No meaningful outliers
# * No missing or duplicate Data
# * Similarity of around 55% (Hamming Distance)

# %%
# Count total number of rows per disease
print("Ploting total number of rows per disease (total_diseases.png):")
print()
count_disease = pd.DataFrame(data["TYPE"]).value_counts().rename_axis('TYPE').reset_index(name='count')
print(count_disease)
plot_st = sb.barplot(data=count_disease, x="count", y="TYPE")
plt.show()


# %%
# Count total number of records per symptom
print("Ploting total number of records per symptom (sum_total.png):")
print()
sum_columns = { "columns" : list(columns.drop("TYPE")), "total" : [data[c].sum() for c in columns if c != 'TYPE']}
sum_columns = pd.DataFrame(sum_columns)
plot_st = sb.barplot(data=sum_columns, x="total", y="columns")
plt.show()

# %% [markdown]
# As we can see, there is some counting similarity between symptoms which are indeed correlated in the real world. 
# 
# Based on that, we decided to order our columns to better visualize correlated symptoms:

# %%
# Order columns and similar symptoms
data = data[[
    'COUGH', 'MUSCLE_ACHES', 'TIREDNESS', 'SORE_THROAT','SNEEZING', 
    'RUNNY_NOSE', 'STUFFY_NOSE', 'LOSS_OF_TASTE', 'LOSS_OF_SMELL', 
    'FEVER',
    'NAUSEA', 'VOMITING', 'DIARRHEA', 'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING', 
    'ITCHY_NOSE', 'ITCHY_EYES', 'ITCHY_MOUTH', 'ITCHY_INNER_EAR', 'PINK_EYE', 
    'TYPE']]

columns = data.columns
symptoms = list(columns)
symptoms.remove("TYPE")
diseases = list(data["TYPE"].unique())


# %%
# Count total number of records per symptom
print("Ploting total number of records per symptom (sum_total.png):")
print()
sum_columns = { "columns" : list(columns.drop("TYPE")), "total" : [data[c].sum() for c in columns if c != 'TYPE']}
sum_columns = pd.DataFrame(sum_columns)
plot_st = sb.barplot(data=sum_columns, x="total", y="columns")
plt.show()


# %%
# Get symptoms correlation factor
print("Ploting symptoms correlation factor (correlation.png):")
print()
correlation = data.drop("TYPE", axis=1)
correlation = correlation.corr(method="spearman", min_periods=44000)
plot_s = sb.heatmap(correlation, annot=False)
plt.show()

# %% [markdown]
# There is also another measure we considered visualizing: 
# * The number of symptoms registered per person.
# 
# As will be shown in the plot, every disease has a similar distribution of the number of symptoms registered per person, being that distribution between minimum 1 and maximum 14 symptoms. This, we concluded, may not say much, because the mean values are very close to each other.

# %%
# Count the number of symptoms registered per disease
print("Ploting number of symptoms registered per disease (total_symptoms.png):")
print()
total_symptoms = data.copy()
total_symptoms["TOTAL"] = total_symptoms.sum(axis=1)
total_symptoms = total_symptoms[["TYPE", "TOTAL"]]

ts = pd.DataFrame(0, index=np.arange(16), columns=diseases)

# Normalize (min-max) the values
for d in diseases:
    n = total_symptoms.loc[total_symptoms["TYPE"] == d]
    occ = n["TOTAL"].value_counts()
    mx = float(occ.max())
    mn = float(occ.min())

    ddt = pd.DataFrame(occ).sort_index(0)

    for i,r in ddt.iterrows():
        old = r["TOTAL"]
        new = float(old - mn) / float(mx-mn)
        ts.loc[i,d] = new

plot_ts = sb.lineplot(data=ts)
plt.show()

# %% [markdown]
# Then, we decided to measure the percentage of people with a given symptom for every disease. The next plot, on the other hand, gives us now a clear perception of the correlation between some symptoms as well as the fact that some groups of symptoms only occur in distinct diseases or patterns.

# %%
# Get the percentage of occurrence of every symptom per desease
print("Ploting percentage of occurrence of every symptom per desease (pair_symptoms.png):")
print()
frequency_symptoms = []
total_diseases = data[["TYPE"]]
total_diseases = pd.DataFrame([total_diseases["TYPE"].value_counts()]).reset_index().drop("index", axis=1)

for s in symptoms:
    frequency_symptoms.append([s] + [0 for i in range(len(diseases))])

for i,r in data.iterrows():
    d = r["TYPE"]
    for i in range(len(symptoms)):
        if r[symptoms[i]] == 1:
            frequency_symptoms[i][diseases.index(d) + 1] += 1

for fq in frequency_symptoms:
    for d in range(len(diseases)):
        total = total_diseases[diseases[d]][0]
        fq[d+1] = int(fq[d+1] / total * 100)

frequency_symptoms = pd.DataFrame(frequency_symptoms, columns=(["SYMPTOM"] + diseases))
plot_ps = sb.catplot(col="SYMPTOM", col_wrap=4,
                data=frequency_symptoms,
                kind="bar", height=2.5, aspect=.8)
plt.show()

# %% [markdown]
# We also applied the *SelectKBest* scikitlearn algorithm with the *chi2* score funtion (better suited for binary data) to find our best features in relation to the solution column. 
# 
# This plot could also confirm the assumptions made in the previous analysis, where we found highly correlated symtoms that could very quickly be used to infer some types of diseases.

# %%
# Univariate Selection to find the best scoring features
# https://en.wikipedia.org/wiki/Dimensionality_reduction
from sklearn.feature_selection import SelectKBest, chi2

fit = SelectKBest(chi2).fit(data[symptoms], y=data['TYPE'])
plot_st = sb.barplot(x=fit.scores_, y=symptoms, orient="h")
plt.show()
plot_st.get_figure().savefig("./out/6_kbest_features.png", dpi=200)
plot_st.get_figure().clf()

# %% [markdown]
# ### Data Preprocessing
# 
# After visualizing the data and making some clear points on the patterns observed by similar symptoms, we decided to apply a set of actions to preprocess our data:
# 
# #### 1. Encode the *class* column 

# %%
from sklearn.preprocessing import LabelEncoder

# Rename Column Type to class
data = data.rename(columns={'TYPE' : 'class'})

# Encode the classes names to representative numbers 
print("Encoding the classes names to representative numbers:")
label_encoder = LabelEncoder()                   
data['class']= label_encoder.fit_transform(data['class'])

# Save encoding values for future reference
keys = label_encoder.classes_
values = label_encoder.transform(keys)
dictionary = dict(zip(keys, values))
print(dictionary)

# %% [markdown]
# #### 2. Aggregate similar symptoms into new columns
# These new columns will be the result of a **Logical OR** applied to the aggregated columns.
# We decided, based on the previous plots, to aggregate the *NAUSEA* & *VOMITING* related symptoms as well as all the *ITCHY* ones into new columns. 
# 
# This action helped us reducing the total number of features from 21 to 13.

# %%
# Aggregate similar symptoms
# New symptom VOMIT = [LOGICAL OR] 'NAUSEA', 'VOMITING', 'DIARRHEA', 'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING'
data['VOMIT'] = data['NAUSEA'] + data['VOMITING'] + data['DIARRHEA'] + data['SHORTNESS_OF_BREATH'] + data['DIFFICULTY_BREATHING']
data['VOMIT'] = data['VOMIT'].apply(lambda x: bool(x))
data = data.drop(['NAUSEA', 'VOMITING', 'DIARRHEA', 'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING'], axis=1)


# New symptom ITCHY = [LOGICAL OR] 'ITCHY_NOSE', 'ITCHY_EYES', 'ITCHY_MOUTH', 'ITCHY_INNER_EAR', 'PINK_EYE'
data['ITCHY'] = data['ITCHY_NOSE'] + data['ITCHY_EYES'] + data['ITCHY_MOUTH'] + data['ITCHY_INNER_EAR'] + data['PINK_EYE']
data['ITCHY'] = data['ITCHY'].apply(lambda x: bool(x))
data = data.drop(['ITCHY_NOSE', 'ITCHY_EYES', 'ITCHY_MOUTH', 'ITCHY_INNER_EAR', 'PINK_EYE'], axis=1)

# %% [markdown]
# After all the data preprocessing, now we are good to go on start splitting our data between two sets:
# * 80% for training
# * 20% for testing

# %%
from sklearn.model_selection import train_test_split

labels = data.pop('class')
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=5)

# %% [markdown]
# ### Algorithms
# Let's start the application of different ML Algorithms to train and create a predictive model for our data. The Algorithms applied will be:
# * Decision Tree
# * Random Forest
# * SVM (Support-Vector Machine)
# * K-NN (K-Nearest Neighbour)
# * Neural Networks
# 
# As tuning approaches we decided to use the scikitlearn functions *StratifiedKFold* and *GridSearchCV* on our training process. The first one is to avoid overfitting and also to balance our database (not all classes have the same number of records) during the fitting process. The last one is used to test different parameters for the classifiers and choosing the best ones. 

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# %% [markdown]
# #### Decision Tree
# We applied the Decision Tree Classifier, before, with a set of different parameters which can be tested replacing the parameter grid in the code with this one:
# ```python
# parameter_grid = {'criterion': ['gini', 'entropy'],
#                   'splitter': ['best', 'random'], 
#                   'max_depth': [1,2,5,10],
#                   'max_features': [5,8,10,12]}
# ```

# %%
from sklearn.tree import DecisionTreeClassifier


decision_tree_classifier = DecisionTreeClassifier()

parameter_grid = {'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random'], 
                  'max_depth': [5],
                  'max_features': [12]}

cross_validation = StratifiedKFold(n_splits=10,  shuffle=True)

grid_search = GridSearchCV(decision_tree_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)
print("DECISION TREE CLASSIFIER:")
print()

start = time.process_time()
grid_search.fit(X_train, y_train)
print('Time Elapsed: {} seconds'.format(time.process_time() - start))

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

y_pred = grid_search.predict(X_test)

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print()
print("Confusion matrix:")
print()
print(dictionary)
cfmx = confusion_matrix(y_test, y_pred)
print(cfmx)
print()
print("Report: ")
print(classification_report(y_test, y_pred))

plot_cfmx = sb.heatmap(cfmx, xticklabels=diseases, yticklabels=diseases, annot=True)
plt.show()

# %% [markdown]
# ![](./out/graphviz.png)
# %% [markdown]
# #### Random Forest
# We also tried the Random Forest Classifier. 
# As this algorithm uses a set of Decision Trees this means that results may be quite similar, except for the processing time. We also tested it before, with a set of different parameters which can be tested replacing the parameter grid in the code with this one:
# ```python
# parameter_grid = {'n_estimators': [2, 5, 10],
#                 'criterion': ['gini', 'entropy'],
#                 'bootstrap' : ["True", "False"],
#                 'max_depth' : [1,2,5],
#                 'max_features': [8,10,12],
#                 'n_jobs' : [5]}
# ```
# 

# %%
from sklearn.ensemble import RandomForestClassifier

parameter_grid = {'n_estimators': [5],
                'criterion': ['gini', 'entropy'],
                'bootstrap' : ["True", "False"],
                'max_depth' : [5],
                'max_features': [12],
                'n_jobs' : [5]
}

random_forest_classifier = RandomForestClassifier()

cross_validation = StratifiedKFold(n_splits=10,  shuffle=True)

grid_search = GridSearchCV(random_forest_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)

print("RANDOM FOREST CLASSIFIER:")
print()

start = time.process_time()
grid_search.fit(X_train, y_train)
print('Time Elapsed: {} seconds'.format(time.process_time() - start))
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

y_pred = grid_search.predict(X_test)

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print()
print("Confusion matrix:")
print()
print(dictionary)
cfmx = confusion_matrix(y_test, y_pred)
print(cfmx)
print()
print("Report: ")
print(classification_report(y_test, y_pred))
plot_cfmx = sb.heatmap(cfmx, xticklabels=diseases, yticklabels=diseases, annot=True)
plt.show()

# %% [markdown]
# #### Support Vector Machine
# SVMs are a discriminative classifier: that is, they draw a boundary between clusters of data. We also tested it with different values for the *kernel* parameter and let the GridSearch to choose the best.
# 

# %%

from sklearn.svm import SVC

classifierSVC = SVC()

svc_parameter_grid = {'kernel': ['linear', 'rbf']}

svc_cross_validation = StratifiedKFold(n_splits=10,  shuffle=True)

grid_search = GridSearchCV(classifierSVC,
                           param_grid=svc_parameter_grid,
                           cv=svc_cross_validation)

print("SVM CLASSIFIER:")
print()
start = time.process_time()
grid_search.fit(X_train, y_train)
print('Time Elapsed: {} seconds'.format(time.process_time() - start))
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

y_pred = grid_search.predict(X_test)

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print()
print("Confusion matrix:")
print()
print(dictionary)
cfmx = confusion_matrix(y_test, y_pred)
print(cfmx)
print()
print("Report: ")
print(classification_report(y_test, y_pred))
plot_cfmx = sb.heatmap(cfmx, xticklabels=diseases, yticklabels=diseases, annot=True)
plt.show()

# %% [markdown]
# #### KNeighbors
# We also tested this one before, with a set of different parameters which can be tested replacing the parameter grid in the code with this one:
# ```python
# knn_parameter_grid = {'n_neighbors': [3,5,10],
#                 'weights': ['uniform’, ‘distance’],
#                 'algorithm': [‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’],
#                 'n_jobs' : [5]
# ```
# 

# %%
from sklearn.neighbors import KNeighborsClassifier

classifierKNN = KNeighborsClassifier()

knn_parameter_grid = {
                'n_neighbors': [5],
                'weights': ['uniform'],
                'algorithm': ['brute'],
                'n_jobs' : [5]
}

knn_cross_validation = StratifiedKFold(n_splits=10,  shuffle=True)

grid_search = GridSearchCV(classifierKNN,
                           param_grid=knn_parameter_grid,
                           cv=knn_cross_validation)

print("K-NN CLASSIFIER:")
print()

start = time.process_time()
grid_search.fit(X_train, y_train)
print('Time Elapsed: {} seconds'.format(time.process_time() - start))
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

y_pred = grid_search.predict(X_test)

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print()
print("Confusion matrix:")
print()
print(dictionary)
cfmx = confusion_matrix(y_test, y_pred)
print(cfmx)
print()
print("Report: ")
print(classification_report(y_test, y_pred))
plot_cfmx = sb.heatmap(cfmx, xticklabels=diseases, yticklabels=diseases, annot=True)
plt.show()

# %% [markdown]
# #### Neural Networks
# Finally, we also applied NN before, with a set of different parameters which can be tested replacing the parameter grid in the code with this one:
# ```python
# parameter_grid_nn = {
#                 'activation': ['relu','tanh'],
#                 'solver': ['sgd', 'adam'],
#                 'hidden_layer_sizes' : [(12,12,12,12),(8,8,8,8,8))]}
# ```
# (3,32,6,32)
# (3,5,12,32)
# 

# %%

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

neural_network_classifier = MLPClassifier()

parameter_grid_nn = {
                'activation': ['tanh'],
                'solver': ['adam'],
                'hidden_layer_sizes' : [(3,5,8,13,21,34)],
                'verbose': [True]}

cross_validation_nn = StratifiedKFold(n_splits=10,  shuffle=True)

grid_search = GridSearchCV(neural_network_classifier,
                           param_grid=parameter_grid_nn,
                           cv=cross_validation_nn)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("NEURAL NETWORK CLASSIFIER:")
print()

start = time.process_time()
grid_search.fit(X_train, y_train)
print('Time Elapsed: {} seconds'.format(time.process_time() - start))
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

y_pred = grid_search.predict(X_test)

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print()
print("Confusion matrix:")
print()
print(dictionary)
cfmx = confusion_matrix(y_test, y_pred)
print(cfmx)
print()
print("Report: ")
print(classification_report(y_test, y_pred))
plot_cfmx = sb.heatmap(cfmx, xticklabels=diseases, yticklabels=diseases, annot=True)
plt.show()


# %%



