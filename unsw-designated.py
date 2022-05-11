import pandas as pd
import numpy as np
from statistics import mean, stdev
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# To UNSW-NB15's authors' credit, they do strip dst port from their DESIGNATED train / test sets, still sttl ruins their entire dataset
head = "/home/dhoogla/PhD/clean-ids-collection"
unsw_data = {    
    "unsw-nb15-train": f"{head}/unsw-nb15/clean/designated-train-test-sets/UNSW_NB15_training-set.parquet",
    "unsw-nb15-test": f"{head}/unsw-nb15/clean/designated-train-test-sets/UNSW_NB15_testing-set.parquet"
}
result_data = []

tr = pd.read_parquet(unsw_data['unsw-nb15-train'])
tr['subset'] = 'train'
print(tr.shape)
tr = tr.drop(index=tr[tr['service'] == '-'].index)
print(tr.shape)
te = pd.read_parquet(unsw_data['unsw-nb15-test'])
te['subset'] = 'test'
print(te.shape)
te = te.drop(index=te[te['service'] == '-'].index)
print(te.shape)
        
df = pd.concat(objs=[tr, te], ignore_index=True, copy=False, sort=False)

df = df[['service', 'subset', 'label']]

attack_types = list(df['label'].value_counts().index)
class_idx = df.columns.size -1
df['label'] = df['label'].astype(dtype='float32', copy=False)

df['service'] = df['service'].astype('category')
df['service'] = df['service'].cat.codes

print(df.dtypes)
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
print(df_train['label'].value_counts())
gp = df_train.groupby('label')
print("TRAINING SET")
with pd.option_context('display.max_rows', 50):
    print(gp.get_group(0.0)['service'].value_counts())
    print(gp.get_group(1.0)['service'].value_counts())


df_test = df.loc[df['subset'] == 'test']
df_test.drop(labels=['subset'], axis=1, inplace=True)
df_test.reset_index(inplace=True, drop=True)
print(df_test.shape)
print(df_test['label'].value_counts())
print("TESTING SET")
gp = df_test.groupby('label')
with pd.option_context('display.max_rows', 50):
    print(gp.get_group(0.0)['service'].value_counts())
    print(gp.get_group(1.0)['service'].value_counts())

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

    ts = 0.8
    X_train, X_test, y_train, y_test = train_test_split(df['service'], df['label'], test_size=ts)
    
    X_train = df_train['service']
    y_train = df_train['label']
    X_test = df_test['service']
    y_test = df_test['label']

    model.fit(X=X_train.array.reshape(-1,1), y=y_train.array)

    intra_outputs = model.predict(X=X_test.array.reshape(-1,1))
    intra_acc = accuracy_score(y_true=y_test, y_pred=intra_outputs)
    intra_rec = recall_score(y_true=y_test, y_pred=intra_outputs)
    intra_pre = precision_score(y_true=y_test, y_pred=intra_outputs)

    accuracies.append(intra_acc)
    recalls.append(intra_rec)
    precisions.append(intra_pre)

result_row = [
    'unsw-nb15-designated',
    -1.0,
    round(mean(accuracies), 3),
    round(stdev(accuracies), 3),
    round(mean(precisions), 3),
    round(stdev(precisions), 3),
    round(mean(recalls), 3),
    round(stdev(recalls), 3)    
]
print('unws-nb15-designated', result_row)
result_data.append(result_row)