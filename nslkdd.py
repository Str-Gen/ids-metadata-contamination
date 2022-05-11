import pandas as pd
import numpy as np
from statistics import mean, stdev
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# NSL-KDD does not directly include the numeric destination port, but it does include 'service'
head = "/home/dhoogla/PhD/clean-ids-collection"
nslkdd_data = {    
    "nslkdd-train": f"{head}/nsl-kdd/dirty-with-metadata/KDDTrain.parquet",
    "nslkdd-test": f"{head}/nsl-kdd/dirty-with-metadata/KDDTest.parquet",        
}
nslkdd_data['nslkdd'] = [nslkdd_data['nslkdd-train'], nslkdd_data['nslkdd-test']]

result_data = []

tr = pd.read_parquet(nslkdd_data['nslkdd-train'])
tr['subset'] = 'train'
te = pd.read_parquet(nslkdd_data['nslkdd-test'])
te['subset'] = 'test'
        
df = pd.concat(objs=[tr, te], ignore_index=True, copy=False, sort=False)

df = df[['service', 'subset', 'class']]

attack_types = list(df['class'].value_counts().index)
class_idx = df.columns.size -1
df['class'] = df['class'].astype('object')

attacks = df.loc[df['class'] != "normal"].index
df.iloc[attacks, class_idx] = 1.0
df.iloc[df.index.difference(attacks), class_idx] = 0.0
df['class'] = df['class'].astype(dtype=np.float32, copy=False)
print(df['class'].value_counts())

df['service'] = df['service'].astype('category')
df['service'] = df['service'].cat.codes

print(df.shape)

col = df.columns[-1]
cols = df.columns[:-1]
vc = df[col].value_counts()
n = vc.iloc[-1]
m = vc.iloc[0]
initial_cut = df.loc[df[col] == vc.index[0]].sample(n=int(m-n), replace=False)
remainder = df.iloc[initial_cut.index, :]    
df = df.drop(index=initial_cut.index)

print(df.shape)

df_train = df.loc[df['subset'] == 'train']
df_train.drop(labels=['subset'], axis=1, inplace=True)    
df_train.reset_index(inplace=True, drop=True)
print(df_train.shape)

df_test = df.loc[df['subset'] == 'test']
df_test.drop(labels=['subset'], axis=1, inplace=True)
df_test.reset_index(inplace=True, drop=True)
print(df_test.shape)

df.drop(labels=['subset'], axis=1, inplace=True)

accuracies = []
accuracies_remainder = []
precisions = [] 
recalls = []

for i in range(10):
    model = RF()
    best_param_dict = {
            'bootstrap': True,
            'max_depth': 16,
            'max_features': 'sqrt',
            'min_impurity_decrease': 0.0,
            'min_samples_leaf': 2,
            'n_estimators': 10,
            'max_samples': 0.5,
            'criterion': 'entropy',
            'n_jobs': -1,
            }
    model.set_params(**best_param_dict)

    ts = 0.2
    X_train, X_test, y_train, y_test = train_test_split(df['service'], df['class'], test_size=ts)
    
    X_train = df_train['service']
    y_train = df_train['class']
    X_test = df_test['service']
    y_test = df_test['class']

    model.fit(X=X_train.array.reshape(-1,1), y=y_train.array)

    intra_outputs = model.predict(X=X_test.array.reshape(-1,1))
    intra_acc = accuracy_score(y_true=y_test, y_pred=intra_outputs)
    intra_rec = recall_score(y_true=y_test, y_pred=intra_outputs)
    intra_pre = precision_score(y_true=y_test, y_pred=intra_outputs)

    accuracies.append(intra_acc)
    recalls.append(intra_rec)
    precisions.append(intra_pre)

result_row = [
    'nsl-kdd-designated',
    -1.0,
    round(mean(accuracies), 3),
    round(stdev(accuracies), 3),
    round(mean(precisions), 3),
    round(stdev(precisions), 3),
    round(mean(recalls), 3),
    round(stdev(recalls), 3)    
]
print('nsl-kdd-designated', result_row)
result_data.append(result_row)