
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


# Get separate list of symptoms and deseases
columns = data.columns
symptoms = list(columns)
symptoms.remove("TYPE")
deseases = list(data["TYPE"].unique())


# Check for columns with not allowed values
print("Checking for columns with not allowed values:")
for i in symptoms:
    values = data[i].unique()
    if not np.array_equiv(np.sort(values), np.array([0,1])):
        print("Column {} has different values: {}".format(i, values))
print()


# Count total number of rows per desease
print("Ploting total number of rows per desease (total_deseases.png)")
count_desease = pd.DataFrame(data["TYPE"]).value_counts().rename_axis('TYPE').reset_index(name='count')
plot_st = sb.barplot(data=count_desease, x="count", y="TYPE")
plot_st.get_figure().savefig("./out/1_total_deseases.png", dpi=200)
plot_st.get_figure().clf()


# Count total number of records per symptom
print("Ploting total number of records per symptom (sum_total.png)")
sum_columns = { "columns" : list(columns.drop("TYPE")) + [ r["TYPE"] for i,r in count_desease.iterrows() ], "total" : [data[c].sum() for c in columns if c != 'TYPE'] + [ r["count"] for i,r in count_desease.iterrows() ]}
sum_columns = pd.DataFrame(sum_columns)
plot_st = sb.barplot(data=sum_columns, x="total", y="columns")
plot_st.get_figure().savefig("./out/2_sum_total.png", dpi=200)
plot_st.get_figure().clf()


# Get symptoms correlation factor
print("Ploting symptoms correlation factor (correlation.png)")
correlation = data.drop("TYPE", axis=1)
correlation = correlation.corr(method="spearman", min_periods=44000)
plot_s = sb.heatmap(correlation, annot=False)    
plot_s.get_figure().savefig("./out/3_correlation.png", dpi=200)
plot_s.get_figure().clf()


# Count the number of symptoms registered per desease
print("Ploting number of symptoms registered per desease (total_symptoms.png)")
total_symptoms = data.copy()
total_symptoms["TOTAL"] = total_symptoms.sum(axis=1)
total_symptoms = total_symptoms[["TYPE", "TOTAL"]]

ts = pd.DataFrame(0, index=np.arange(16), columns=deseases)

# Normalize (min-max) the values
for d in deseases:
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
plot_ts.get_figure().savefig("./out/4_total_symptoms.png", dpi=200)
plot_ts.get_figure().clf()


# Get percentage of occurrence of every symptom per desease
print("Ploting percentage of occurrence of every symptom per desease (pair_symptoms.png)")
frequency_symptoms = []
total_deseases = data[["TYPE"]]
total_deseases = pd.DataFrame([total_deseases["TYPE"].value_counts()]).reset_index().drop("index", axis=1)

for s in symptoms:
    frequency_symptoms.append([s] + [0 for i in range(len(deseases))])

for i,r in data.iterrows():
    d = r["TYPE"]
    for i in range(len(symptoms)):
        if r[symptoms[i]] == 1:
            frequency_symptoms[i][deseases.index(d) + 1] += 1

for fq in frequency_symptoms:
    for d in range(len(deseases)):
        total = total_deseases[deseases[d]][0]
        fq[d+1] = int(fq[d+1] / total * 100)

# frequency_symptoms = pd.DataFrame(frequency_symptoms, columns=(["SYMPTOM"] + deseases))
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


# Generate a new column based on the total number of symptoms registered per row
data_total = data[symptoms].sum(axis=1)
mn = data_total.min()
mx = data_total.max()
data_total = data_total.apply(lambda x : float(x - mn) / float(mx-mn))
# Append previously generated data['TOTAL'] column
data['TOTAL'] = data_total


# Univariate Selection to find the best scoring features
# https://en.wikipedia.org/wiki/Dimensionality_reduction
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

fit = SelectKBest().fit(data[symptoms + ["TOTAL"]], y=data['class'])
plot_st = sb.barplot(x=fit.scores_, y=symptoms + ["TOTAL"], orient="h")
plot_st.get_figure().savefig("./out/6_kbest_features.png", dpi=200)
plot_st.get_figure().clf()

pca = PCA()
pca = pca.fit(data[symptoms + ["TOTAL"]], y=data['class'])
plot_st = sb.barplot(x=pca.explained_variance_ratio_, y=symptoms + ["TOTAL"], orient="h")
plot_st.get_figure().savefig("./out/7_pca.png", dpi=200)
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


# Get new symptoms correlation factor
print("Ploting symptoms correlation factor (correlation_2.png)")
correlation = data.drop("class", axis=1)
correlation = correlation.corr(method="spearman", min_periods=44000)
plot_s = sb.heatmap(correlation, annot=True)    
plot_s.get_figure().savefig("./out/8_correlation_2.png", dpi=200)
plot_s.get_figure().clf()
print(correlation)
print()


new_columns = list(data.columns)
new_columns.remove("class")
fit = SelectKBest().fit(data[new_columns], y=data['class'])
plot_st = sb.barplot(x=fit.scores_, y=new_columns, orient="h")
plot_st.get_figure().savefig("./out/9_kbest_features_2.png", dpi=200)
plot_st.get_figure().clf()


pca = PCA()
pca = pca.fit(data[new_columns], y=data['class'])
plot_st = sb.barplot(x=pca.explained_variance_ratio_, y=new_columns, orient="h")
plot_st.get_figure().savefig("./out/10_pca_2.png", dpi=200)
plot_st.get_figure().clf()