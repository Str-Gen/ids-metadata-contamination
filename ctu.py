import pandas as pd
import numpy as np
from statistics import mean, stdev
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

head = "/home/dhoogla/PhD/clean-ids-collection"
ctu_data = {    
    'ctu13-10': f'{head}/ctu-13/dirty-with-metadata/10/capture20110818.binetflow.parquet',
    'ctu13-11': f'{head}/ctu-13/dirty-with-metadata/11/capture20110818-2.binetflow.parquet',
    'ctu13-12': f'{head}/ctu-13/dirty-with-metadata/12/capture20110819.binetflow.parquet',
    'ctu13-13': f'{head}/ctu-13/dirty-with-metadata/13/capture20110815-3.binetflow.parquet',
    'ctu13-1': f'{head}/ctu-13/dirty-with-metadata/1/capture20110810.binetflow.parquet',
    'ctu13-2': f'{head}/ctu-13/dirty-with-metadata/2/capture20110811.binetflow.parquet',
    'ctu13-3': f'{head}/ctu-13/dirty-with-metadata/3/capture20110812.binetflow.parquet',
    'ctu13-4': f'{head}/ctu-13/dirty-with-metadata/4/capture20110815.binetflow.parquet',
    'ctu13-5': f'{head}/ctu-13/dirty-with-metadata/5/capture20110815-2.binetflow.parquet',
    'ctu13-6': f'{head}/ctu-13/dirty-with-metadata/6/capture20110816.binetflow.parquet',
    'ctu13-7': f'{head}/ctu-13/dirty-with-metadata/7/capture20110816-2.binetflow.parquet',
    'ctu13-8': f'{head}/ctu-13/dirty-with-metadata/8/capture20110816-3.binetflow.parquet',
    'ctu13-9': f'{head}/ctu-13/dirty-with-metadata/9/capture20110817.binetflow.parquet',
    'ctu13': f'{head}/ctu-13/dirty-with-metadata/all/ctu-13.binetflow.parquet'   
}

# Botnets
# Neris 1-2-9
# Rbot 3-4-10-11
# Virut 5-13
# Menti 6
# Sogou 7
# Murlo 8
# NSIS.ay 12

result_data = []
# Metadata features srcaddr, sport, dstaddr, dport, proto, starttime
metadata_feature = 'starttime'

for k,v in ctu_data.items():
    dataset = k   
    if isinstance(v, list):
        datapaths = v        
    else:
        datapaths = [v]
        
    df = pd.concat(objs=[pd.read_parquet(path) for path in datapaths], ignore_index=True, copy=False, sort=False, verify_integrity=False)
    
    try:
        df = df[[metadata_feature, 'label']]
    except KeyError:
        print(f'{metadata_feature} is not in {k}')
        continue
    
    if metadata_feature == 'srcaddr' or metadata_feature == 'dstaddr' or metadata_feature == 'proto':
        df[metadata_feature] = df[metadata_feature].astype('category').cat.codes    
    elif metadata_feature == 'starttime':
        if df[metadata_feature].dtype == 'object':        
            df[metadata_feature] = pd.to_datetime(df[metadata_feature])        
        df[metadata_feature] = (df[metadata_feature] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    

    df['label'] = df['label'].astype(dtype='object')
    df['label'].value_counts()

    df['label'] = df['label'].str.startswith('flow=From-Botnet', na=False)
    df['label'] = df['label'].astype(dtype='float32', copy=False)

    col = df.columns[-1]
    cols = df.columns[:-1]
    vc = df[col].value_counts()
    n = vc.iloc[-1]
    m = vc.iloc[0]
    if n == m or n < 250 or m < 250:
        continue
    initial_cut = df.loc[df[col] == vc.index[0]].sample(n=int(m-n), replace=False)

    remainder = df.iloc[initial_cut.index, :]    
    df = df.drop(index=initial_cut.index) 

    accuracies = []
    accuracies_remainder = []
    precisions = [] 
    recalls = []

    for i in range(10):
        model = XGBClassifier()
        best_param_dict = {
                # RF
                # 'bootstrap': True,
                # 'max_depth': 16,
                # 'max_features': 'sqrt',
                # 'min_impurity_decrease': 0.0,
                # 'min_samples_leaf': 2,
                # 'n_estimators': 10,
                # 'max_samples': 0.5,
                # 'criterion': 'entropy',
                # 'n_jobs': -1,
                # XGB
                'n_estimators': 10,
                'use_label_encoder': False,
                'max_depth': 6,
                'booster': 'gbtree',
                'tree_method': 'gpu_hist',
                'subsample': 0.5,
                'colsample_bytree': 0.5,
                'importance_type': 'gain',
                'eval_metric': 'logloss',
                'gpu_id': 0,
                'predictor': 'gpu_predictor',                
                'n_jobs': 8, 
            }
        model.set_params(**best_param_dict)

        ts = 0.2
        X_train, X_test, y_train, y_test = train_test_split(df[metadata_feature], df['label'], test_size=ts)

        X_remainder = remainder[metadata_feature]
        y_remainder = remainder['label']

        model.fit(X=X_train.array.reshape(-1,1), y=y_train.array)

        intra_outputs = model.predict(X=X_test.array.reshape(-1,1))
        intra_acc = accuracy_score(y_true=y_test, y_pred=intra_outputs)
        intra_rec = recall_score(y_true=y_test, y_pred=intra_outputs)
        intra_pre = precision_score(y_true=y_test, y_pred=intra_outputs)

        remainder_outputs = model.predict(X=X_remainder.array.reshape(-1, 1))        
        remainder_acc = accuracy_score(y_true=y_remainder, y_pred=remainder_outputs)        
        
        accuracies.append(intra_acc)
        recalls.append(intra_rec)
        precisions.append(intra_pre)
        accuracies_remainder.append(remainder_acc)

    result_row = [
        dataset,
        ts,
        round(mean(accuracies), 3),
        round(stdev(accuracies), 3),
        round(mean(precisions), 3),
        round(stdev(precisions), 3),
        round(mean(recalls), 3),
        round(stdev(recalls), 3),
        round(mean(accuracies_remainder), 3),
        round(stdev(accuracies_remainder), 3)
    ]
    print(dataset, result_row)
    result_data.append(result_row)

result_df = pd.DataFrame(data=result_data, columns=['dataset', 'test_size', 'accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std', 'recall_mean', 'recall_std', 'accuracy_remainder_mean', 'accuracy_remainder_std'])
result_df.to_csv(f'results/ctu-{metadata_feature}.csv', header=True, index=False, encoding='utf-8')
print(result_df)
    
    




