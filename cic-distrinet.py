import pandas as pd
import numpy as np
from statistics import mean, stdev
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

head = "/home/dhoogla/PhD/clean-ids-collection"
cic_data = {
    # "cicids2017-benign": f'{head}/cic-ids2017-distrinet/dirty-with-metadata/Benign-Monday-WorkingHours.parquet',
    "cicids2017-botnet": f'{head}/cic-ids2017-distrinet/dirty-with-metadata/Portscan-DDoS-Botnet-Friday-WorkingHours.parquet',
    "cicids2017-bruteforce": f'{head}/cic-ids2017-distrinet/dirty-with-metadata/Bruteforce-Tuesday-WorkingHours.parquet',
    "cicids2017-ddos": f'{head}/cic-ids2017-distrinet/dirty-with-metadata/Portscan-DDoS-Botnet-Friday-WorkingHours.parquet',
    "cicids2017-dos": f'{head}/cic-ids2017-distrinet/dirty-with-metadata/DoS-Wednesday-WorkingHours.parquet',
    "cicids2017-infiltration": f'{head}/cic-ids2017-distrinet/dirty-with-metadata/Infiltration-WebAttacks-Thursday-WorkingHours.parquet',
    "cicids2017-portscan": f'{head}/cic-ids2017-distrinet/dirty-with-metadata/Portscan-DDoS-Botnet-Friday-WorkingHours.parquet',
    "cicids2017-webattacks": f'{head}/cic-ids2017-distrinet/dirty-with-metadata/Infiltration-WebAttacks-Thursday-WorkingHours.parquet',
    "cicids2017": f'{head}/cic-ids2017-distrinet/dirty-with-metadata/cicids2017-distrinet.parquet',
}

result_data = []
# Metadata features Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp
metadata_feature = 'Timestamp'

for k,v in cic_data.items():    
    dataset = k   
    if isinstance(v, list):
        datapaths = v        
    else:
        datapaths = [v]
        
    df = pd.concat(objs=[pd.read_parquet(path) for path in datapaths], ignore_index=True, copy=False, sort=False, verify_integrity=False)
    try:
        df = df[[metadata_feature, 'Label']]
    except KeyError:
        print(f'{metadata_feature} is not in {k}')
        continue    

    if metadata_feature == 'Source IP' or metadata_feature == 'Destination IP':
        df[metadata_feature] = df[metadata_feature].astype('category').cat.codes    
    elif metadata_feature == 'Timestamp':
        if df[metadata_feature].dtype == 'object':        
            df[metadata_feature] = pd.to_datetime(df[metadata_feature])        
        df[metadata_feature] = (df[metadata_feature] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    df['Label'] = df['Label'].astype(dtype='object')
    print(df['Label'].value_counts())

    if 'ddos' in k:
        df = df[df['Label'].isin(['Benign', 'DDoS'])]
    elif 'botnet' in k:
        df = df[df['Label'].isin(['Benign', 'Bot'])]
    elif 'portscan' in k:
        df = df[df['Label'].isin(['Benign', 'PortScan'])]
    elif 'webattacks' in k:
        df = df[df['Label'].isin(['Benign', 'Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - Sql Injection'])]
    elif 'infiltration' in k:
        df = df[df['Label'].isin(['Benign', 'Infiltration'])]
    print(df['Label'].value_counts())
    df.reset_index(drop=True, inplace=True) 

    label_idx = df.columns.size -1
    attacks = df.loc[df['Label'] != "Benign"].index
    df.iloc[attacks, label_idx] = 1.0
    df.iloc[df.index.difference(attacks), label_idx] = 0.0
    df['Label'] = df['Label'].astype(dtype='float32', copy=False)

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
        X_train, X_test, y_train, y_test = train_test_split(df[metadata_feature], df['Label'], test_size=ts)

        X_remainder = remainder[metadata_feature]
        y_remainder = remainder['Label']

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
result_df.to_csv(f'all_metadata_results/cic2017-distrinet-{metadata_feature}.csv', header=True, index=False, encoding='utf-8')
print(result_df)
    
    




