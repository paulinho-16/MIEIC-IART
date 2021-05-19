#### IMPORT LIBRARIES

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sb
import os.path

# Load csv data
data = pd.read_csv("../data/large_data.csv")


# Print head values and summary statistics
print(data.head())
print()
print(data.describe())
print()


# Order columns and aggregate similar symptoms
data = data[[
    'COUGH', 'MUSCLE_ACHES', 'TIREDNESS', 'SORE_THROAT','SNEEZING', 
    'RUNNY_NOSE', 'STUFFY_NOSE', 'LOSS_OF_TASTE', 'LOSS_OF_SMELL', 
    'FEVER',
    'NAUSEA', 'VOMITING', 'DIARRHEA', 'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING', 
    'ITCHY_NOSE', 'ITCHY_EYES', 'ITCHY_MOUTH', 'ITCHY_INNER_EAR', 'PINK_EYE', 
    'TYPE']]


# Get separate list of symptoms and diseases
columns = data.columns
symptoms = list(columns)
symptoms.remove("TYPE")
diseases = list(data["TYPE"].unique())


# # Check for columns with not allowed values
# print("Checking for columns with not allowed values:")
# for i in symptoms:
#     values = data[i].unique()
#     if not np.array_equiv(np.sort(values), np.array([0,1])):
#         print("Column {} has different values: {}".format(i, values))
# print()


# # Count total number of rows per desease
# print("Ploting total number of rows per desease (total_diseases.png)")
# count_desease = pd.DataFrame(data["TYPE"]).value_counts().rename_axis('TYPE').reset_index(name='count')
# plot_st = sb.barplot(data=count_desease, x="count", y="TYPE")
# plot_st.get_figure().savefig("./out/1_total_diseases.png", dpi=200)
# plot_st.get_figure().clf()


# # Count total number of records per symptom
# print("Ploting total number of records per symptom (sum_total.png)")
# sum_columns = { "columns" : list(columns.drop("TYPE")), "total" : [data[c].sum() for c in columns if c != 'TYPE']}
# sum_columns = pd.DataFrame(sum_columns)
# plot_st = sb.barplot(data=sum_columns, x="total", y="columns")
# plot_st.get_figure().savefig("./out/2_sum_total.png", dpi=200)
# plot_st.get_figure().clf()


# # Get symptoms correlation factor
# print("Ploting symptoms correlation factor (correlation.png)")
# correlation = data.drop("TYPE", axis=1)
# correlation = correlation.corr(method="spearman", min_periods=44000)
# plot_s = sb.heatmap(correlation, annot=False)    
# plot_s.get_figure().savefig("./out/3_correlation.png", dpi=200)
# plot_s.get_figure().clf()


# # Count the number of symptoms registered per desease
# print("Ploting number of symptoms registered per desease (total_symptoms.png)")
# total_symptoms = data.copy()
# total_symptoms["TOTAL"] = total_symptoms.sum(axis=1)
# total_symptoms = total_symptoms[["TYPE", "TOTAL"]]

# ts = pd.DataFrame(0, index=np.arange(16), columns=diseases)

# # Normalize (min-max) the values
# for d in diseases:
#     n = total_symptoms.loc[total_symptoms["TYPE"] == d]
#     occ = n["TOTAL"].value_counts()
#     mx = float(occ.max())
#     mn = float(occ.min())

#     ddt = pd.DataFrame(occ).sort_index(0)

#     for i,r in ddt.iterrows():
#         old = r["TOTAL"]
#         new = float(old - mn) / float(mx-mn)
#         ts.loc[i,d] = new

# plot_ts = sb.lineplot(data=ts)
# plot_ts.get_figure().savefig("./out/4_total_symptoms.png", dpi=200)
# plot_ts.get_figure().clf()


# # Get percentage of occurrence of every symptom per desease
# print("Ploting percentage of occurrence of every symptom per desease (pair_symptoms.png)")
# frequency_symptoms = []
# total_diseases = data[["TYPE"]]
# total_diseases = pd.DataFrame([total_diseases["TYPE"].value_counts()]).reset_index().drop("index", axis=1)

# for s in symptoms:
#     frequency_symptoms.append([s] + [0 for i in range(len(diseases))])

# for i,r in data.iterrows():
#     d = r["TYPE"]
#     for i in range(len(symptoms)):
#         if r[symptoms[i]] == 1:
#             frequency_symptoms[i][diseases.index(d) + 1] += 1

# for fq in frequency_symptoms:
#     for d in range(len(diseases)):
#         total = total_diseases[diseases[d]][0]
#         fq[d+1] = int(fq[d+1] / total * 100)

# frequency_symptoms = pd.DataFrame(frequency_symptoms, columns=(["SYMPTOM"] + diseases))
# plot_ps = sb.catplot(col="SYMPTOM", col_wrap=4,
#                 data=frequency_symptoms,
#                 kind="bar", height=2.5, aspect=.8)
# plot_ps.savefig("./out/5_pair_symptoms.png", dpi=200)


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
print()


# Print head values and summary statistics
print(data.head())
print()
print(data.describe())
print()


# from sklearn.neighbors import DistanceMetric

# dist = DistanceMetric.get_metric('matching')
# np.set_printoptions(suppress=True, precision=2)
# sample = data.sample(500)
# similarity = dist.pairwise(sample)
# print("MAX VALUE {}\nMIN VALUE {}\nMEAN {}".format(np.amax(similarity), np.min(similarity[np.nonzero(similarity)]), np.mean(similarity)))
# np.savetxt("./out/similarity.csv", np.around(similarity, 2), delimiter=",")


# Univariate Selection to find the best scoring features
# https://en.wikipedia.org/wiki/Dimensionality_reduction
from sklearn.feature_selection import SelectKBest, chi2

fit = SelectKBest(chi2).fit(data[symptoms], y=data['class'])
plot_st = sb.barplot(x=fit.scores_, y=symptoms, orient="h")
plot_st.get_figure().savefig("./out/6_kbest_features.png", dpi=200)
plot_st.get_figure().clf()


# Aggregate similar symptoms
# New symptom = [LOGICAL OR] 'NAUSEA', 'VOMITING', 'DIARRHEA', 'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING'
data['VOMIT'] = data['NAUSEA'] + data['VOMITING'] + data['DIARRHEA'] + data['SHORTNESS_OF_BREATH'] + data['DIFFICULTY_BREATHING']
data['VOMIT'] = data['VOMIT'].apply(lambda x: bool(x))
data = data.drop(['NAUSEA', 'VOMITING', 'DIARRHEA', 'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING'], axis=1)


# New symptom ITCHY = [LOGICAL OR] 'ITCHY_NOSE', 'ITCHY_EYES', 'ITCHY_MOUTH', 'ITCHY_INNER_EAR', 'PINK_EYE'
data['ITCHY'] = data['ITCHY_NOSE'] + data['ITCHY_EYES'] + data['ITCHY_MOUTH'] + data['ITCHY_INNER_EAR'] + data['PINK_EYE']
data['ITCHY'] = data['ITCHY'].apply(lambda x: bool(x))
data = data.drop(['ITCHY_NOSE', 'ITCHY_EYES', 'ITCHY_MOUTH', 'ITCHY_INNER_EAR', 'PINK_EYE'], axis=1)

#####################################################################################

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, make_scorer, f1_score, accuracy_score, recall_score, confusion_matrix, classification_report


symptoms = data.columns
symptoms.drop("class")
all_inputs = data[symptoms].values
all_labels = data['class'].values

labels = data.pop('class')
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=5)

# decision_tree_classifier = DecisionTreeClassifier()

# parameter_grid = {'criterion': ['gini', 'entropy'],
#                   'splitter': ['best', 'random'], 
#                   'max_depth': [5],
#                   'max_features': [12]}

# cross_validation = StratifiedKFold(n_splits=10,  shuffle=True)

# grid_search = GridSearchCV(decision_tree_classifier,
#                            param_grid=parameter_grid,
#                            cv=cross_validation)

# grid_search.fit(X_train, y_train)
# print()
# print("DECISION TREE CLASSIFIER:")
# print()
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))

# y_pred = grid_search.predict(X_test)

# print()
# print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
# print()
# print("Confusion matrix:")
# print()
# print(dictionary)
# print(confusion_matrix(y_test, y_pred))
# print()
# print("Report: ")
# print(classification_report(y_test, y_pred))

# import sklearn.tree as tree

# decision_tree_classifier = grid_search.best_estimator_
# with open('./out/covid.dot', 'w') as out_file:
#     out_file = tree.export_graphviz(decision_tree_classifier, out_file=out_file)

# #####################################################################################
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.utils._joblib import joblib
# import sklearn

# parameter_grid = {'n_estimators': [5],
#                 'criterion': ['gini', 'entropy'],
#                 'bootstrap' : ["True", "False"],
#                 'max_depth' : [5],
#                 'max_features': [12],
#                 'n_jobs' : [5]
# }

# random_forest_classifier = RandomForestClassifier()

# cross_validation = StratifiedKFold(n_splits=10,  shuffle=True)

# grid_search = GridSearchCV(random_forest_classifier,
#                            param_grid=parameter_grid,
#                            cv=cross_validation)

# grid_search.fit(X_train, y_train)
# print()
# print("RANDOM FOREST CLASSIFIER:")
# print()
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))

# y_pred = grid_search.predict(X_test)

# print()
# print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
# print()
# print("Confusion matrix:")
# print()
# print(dictionary)
# print(confusion_matrix(y_test, y_pred))
# print()
# print("Report: ")
# print(classification_report(y_test, y_pred))

# #######################################################################################

# from sklearn.svm import SVC

# classifierSVC = SVC()

# svc_parameter_grid = {'kernel': ['linear', 'rbf']}

# svc_cross_validation = StratifiedKFold(n_splits=10,  shuffle=True)

# grid_search_svc = GridSearchCV(classifierSVC,
#                            param_grid=svc_parameter_grid,
#                            cv=svc_cross_validation)

# grid_search_svc.fit(X_train, y_train)

# print()
# print("SVC CLASSIFIER:")
# print()
# print('Best score: {}'.format(grid_search_svc.best_score_))
# print('Best parameters: {}'.format(grid_search_svc.best_params_))

# y_pred = grid_search_svc.predict(X_test)

# print()
# print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
# print()
# print("Confusion matrix:")
# print()
# print(dictionary)
# print(confusion_matrix(y_test, y_pred))
# print()
# print("Report: ")
# print(classification_report(y_test, y_pred))

# #######################################################################################

# from sklearn.neighbors import KNeighborsClassifier

# classifierKNN = KNeighborsClassifier()

# knn_parameter_grid = {
#                 'n_neighbors': [5],
#                 'weights': ['uniform'],
#                 'algorithm': ['brute'],
#                 'n_jobs' : [5]
# }

# knn_cross_validation = StratifiedKFold(n_splits=10,  shuffle=True)

# grid_search_knn = GridSearchCV(classifierKNN,
#                            param_grid=knn_parameter_grid,
#                            cv=knn_cross_validation)

# grid_search_knn.fit(X_train, y_train)

# print()
# print("K-NN CLASSIFIER:")
# print()
# print('Best score: {}'.format(grid_search_knn.best_score_))
# print('Best parameters: {}'.format(grid_search_knn.best_params_))

# y_pred = grid_search_knn.predict(X_test)

# print()
# print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
# print()
# print("Confusion matrix:")
# print()
# print(dictionary)
# print(confusion_matrix(y_test, y_pred))
# print()
# print("Report: ")
# print(classification_report(y_test, y_pred))

#######################################################################################

# import keras 

# # neural network classifier with 2 layers having 32 neurons each
# m=X_train.shape[0]
# n=X_train.shape[1]
# classes=4
# # Create layers
# inputs = keras.layers.Input(shape=(n,), dtype='float32', name='input_layer') # Input (2 dimensions)
# outputs = keras.layers.Dense(32, activation='tanh', name='hidden_layer1')(inputs) # Hidden layer
# outputs = keras.layers.Dense(32, activation='tanh', name='hidden_layer2')(outputs) # Hidden layer
# outputs = keras.layers.Dense(classes, activation='softmax', name='output_layer')(outputs) # Output layer 
# # Create a model from input layer and output layers
# neural_network = keras.models.Model(inputs=inputs, outputs=outputs, name='neural_network')
# # Convert labels to categorical: categorical_crossentropy expects targets to be binary matrices (1s and 0s) of shape (samples, classes)
# neural_network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Y_binary = keras.utils.to_categorical(y_train, num_classes=classes, dtype='int')
# # Train the model on the train set (output debug information)
# neural_network.fit(X_train, Y_binary, epochs=100, verbose=1)

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

neural_network_classifier = MLPClassifier()

parameter_grid_nn = {
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],
                'hidden_layer_sizes' : [(50,50), (32,32,32)]}

cross_validation_nn = StratifiedKFold(n_splits=10,  shuffle=True)

grid_search_neural_network = GridSearchCV(neural_network_classifier,
                           param_grid=parameter_grid_nn,
                           cv=cross_validation_nn)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

grid_search_neural_network.fit(X_train, y_train)

print()
print("NEURAL NETWORK CLASSIFIER:")
print()
print('Best score: {}'.format(grid_search_neural_network.best_score_))
print('Best parameters: {}'.format(grid_search_neural_network.best_params_))

y_pred = grid_search_neural_network.predict(X_test)

print()
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print()
print("Confusion matrix:")
print()
print(dictionary)
print(confusion_matrix(y_test, y_pred))
print()
print("Report: ")
print(classification_report(y_test, y_pred))