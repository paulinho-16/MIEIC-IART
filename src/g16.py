
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

#### LOAD CSV DATA

data = pd.read_csv("../data/large_data.csv")

# print(data.head())
# print()

# print(data.describe())
# print()

columns = data.columns
symptoms = list(columns)
symptoms.remove("TYPE")
deseases = list(data["TYPE"].unique())

# for i in symptoms:
#     values = data[i].unique()
#     if not np.array_equiv(np.sort(values), np.array([0,1])):
#         print("Column {} has different values: {}".format(i, values))

# print()


similarity = data.drop("TYPE", axis=1)
similarity = similarity.corr(method="spearman", min_periods=44000)
plot_s = sb.heatmap(similarity, annot=False)    
plot_s.get_figure().savefig("./out/similarity.png", dpi=200)
plot_s.get_figure().clf()


sum_columns = { "columns" : list(columns.drop("TYPE")), "total" : [data[c].sum() for c in columns if c != 'TYPE'] }
sum_columns = pd.DataFrame(sum_columns)
plot_st = sb.barplot(data=sum_columns, x="total", y="columns")
plot_st.get_figure().savefig("./out/sum_total.png", dpi=200)
plot_st.get_figure().clf()


total_symptoms = data.copy()
total_symptoms["TOTAL"] = total_symptoms.sum(axis=1)
total_symptoms = total_symptoms[["TYPE", "TOTAL"]]
# for d in deseases:
#     n = total_symptoms.loc[total_symptoms["TYPE"] == d]
#     mean = float(n.mean())
#     std = float(n.std())

#     for i,r in n.iterrows():
#         old = total_symptoms.iat[i,1]
#         new = int(float(old - mean) / std * 100)
#         total_symptoms.iat[i,1] = new
plot_ts = sb.displot(data=total_symptoms, x="TOTAL", hue="TYPE", binwidth=1, fill=False, element="step")
plot_ts.savefig("./out/total_symptoms.png", dpi=200)


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

frequency_symptoms = pd.DataFrame(frequency_symptoms, columns=(["SYMPTOM"] + deseases))
plot_ps = sb.catplot(col="SYMPTOM", col_wrap=4,
                data=frequency_symptoms,
                kind="bar", height=2.5, aspect=.8)
plot_ps.savefig("./out/pair_symptoms.png", dpi=200)


# scaler = StandardScaler()

# data = data.rename(columns={'TYPE' : 'class'}) 
# subdata = data[["COUGH", "MUSCLE_ACHES", "TIREDNESS", "SORE_THROAT"]]
# scaler.fit(subdata)
# normalized = scaler.transform(subdata)
# new_data = pd.DataFrame(normalized, columns=["COUGH", "MUSCLE_ACHES", "TIREDNESS", "SORE_THROAT"])

# plot = sb.pairplot(new_data.join(data['class']), hue="class")

# plot.savefig("./out/plot1.png")
# print("################################")

# # label_encoder = LabelEncoder()                               
# # data['class']= label_encoder.fit_transform(data['class']) #performing label encoding in the given column
# # labels = data.pop('class')

# # keys = label_encoder.classes_
# # values = label_encoder.transform(keys)
# # dictionary = dict(zip(keys, values)) #storing the converted column entries as (key,value) pairs

# # print(dictionary)

# # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20,random_state=5)  #splitting the dataset into train and test set

# # scaler = StandardScaler()
# # scaler.fit(X_train)

# # X_train = scaler.transform(X_train)
# # X_test = scaler.transform(X_test)

# # print("################################")

# # print(X_test)