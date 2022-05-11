import pandas as pd
import numpy as np
from statistics import mean, stdev
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

head = "/home/dhoogla/PhD/clean-ids-collection"
cidds_data = {
    "cidds-001-external-1": f"{head}/cidds-001/dirty-with-metadata/traffic/ExternalServer/CIDDS-001-external-week1.parquet", # only benign
    "cidds-001-external-2": f"{head}/cidds-001/dirty-with-metadata/traffic/ExternalServer/CIDDS-001-external-week2.parquet", # Portscan + bruteforce
    "cidds-001-external-3": f"{head}/cidds-001/dirty-with-metadata/traffic/ExternalServer/CIDDS-001-external-week3.parquet", # Portscan + bruteforce
    "cidds-001-external-4": f"{head}/cidds-001/dirty-with-metadata/traffic/ExternalServer/CIDDS-001-external-week4.parquet", # Portscan + bruteforce
    "cidds-001-internal-1": f"{head}/cidds-001/dirty-with-metadata/traffic/OpenStack/CIDDS-001-internal-week1.parquet", # Portscan + pingscan + DoS + bruteforce
    "cidds-001-internal-2": f"{head}/cidds-001/dirty-with-metadata/traffic/OpenStack/CIDDS-001-internal-week2.parquet", # Portscan + pingscan + DoS + bruteforce
    "cidds-001-internal-3": f"{head}/cidds-001/dirty-with-metadata/traffic/OpenStack/CIDDS-001-internal-week3.parquet", # only benign
    "cidds-001-internal-4": f"{head}/cidds-001/dirty-with-metadata/traffic/OpenStack/CIDDS-001-internal-week4.parquet", # only benign
    "cidds-002-internal-1": f"{head}/cidds-002/dirty-with-metadata/traffic/week1.parquet", # Portscan
    "cidds-002-internal-2": f"{head}/cidds-002/dirty-with-metadata/traffic/week2.parquet", # Portscan
}
cidds_data["cidds-001-internal"] = [cidds_data["cidds-001-internal-1"], cidds_data["cidds-001-internal-2"], cidds_data["cidds-001-internal-3"], cidds_data["cidds-001-internal-4"]]
cidds_data["cidds-001-external"] = [cidds_data["cidds-001-external-1"], cidds_data["cidds-001-external-2"], cidds_data["cidds-001-external-3"], cidds_data["cidds-001-external-4"]]
cidds_data["cidds-001"] = [*cidds_data["cidds-001-internal"], *cidds_data["cidds-001-external"]]
cidds_data["cidds-002"] = [cidds_data["cidds-002-internal-1"], cidds_data["cidds-002-internal-2"]]

result_data = []
# Metadata features src_ip_addr, src_pt, dst_ip_addr, dst_pt, date_first_seen
metadata_feature = 'date_first_seen'

for k,v in cidds_data.items():
    dataset = k   
    if isinstance(v, list):
        datapaths = v
    else:
        datapaths = [v]
        
    df = pd.concat(objs=[pd.read_parquet(path) for path in datapaths], ignore_index=True, copy=False, sort=False, verify_integrity=False)
    try:
        df = df[[metadata_feature, 'attack_type']]
    except KeyError:
        print(f'{metadata_feature} is not in {k}')
        continue

    if metadata_feature == 'src_ip_addr' or metadata_feature == 'dst_ip_addr':
        df[metadata_feature] = df[metadata_feature].astype('category').cat.codes    
    elif metadata_feature == 'date_first_seen':
        if df[metadata_feature].dtype == 'object':        
            df[metadata_feature] = pd.to_datetime(df[metadata_feature])        
        df[metadata_feature] = (df[metadata_feature] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    
    df['attack_type'] = df['attack_type'].astype(dtype='object')
    df['attack_type'].value_counts()

    label_idx = df.columns.size -1
    attacks = df.loc[df['attack_type'] != "benign"].index
    df.iloc[attacks, label_idx] = 1.0
    df.iloc[df.index.difference(attacks), label_idx] = 0.0
    df['attack_type'] = df['attack_type'].astype(dtype='float32', copy=False)

    col = df.columns[-1]
    cols = df.columns[:-1]
    vc = df[col].value_counts()
    n = vc.iloc[-1]
    m = vc.iloc[0]
    if n == m:
        continue
    initial_cut = df.loc[df[col] == vc.index[0]].sample(n=int(m-n), replace=False)

    remainder = df.iloc[initial_cut.index, :]    
    df = df.drop(index=initial_cut.index)
    
    # df.shape
    # df['attack_type'].value_counts()


    # gp = df.groupby('attack_type')
    # with pd.option_context('display.max_rows', 50):
    #     print(gp.get_group(0.0)['dst_pt'].value_counts())
    #     print(gp.get_group(1.0)['dst_pt'].value_counts())

    # hist = df.hist(column='dst_pt', by='attack_type', bins=1024)
        
        
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
        X_train, X_test, y_train, y_test = train_test_split(df[metadata_feature], df['attack_type'], test_size=ts)

        X_remainder = remainder[metadata_feature]
        y_remainder = remainder['attack_type']

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
result_df.to_csv(f'results/cidds-{metadata_feature}.csv', header=True, index=False, encoding='utf-8')
print(result_df)
