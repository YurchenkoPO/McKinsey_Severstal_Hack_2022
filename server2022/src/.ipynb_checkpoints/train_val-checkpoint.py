import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import shap

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from catboost import Pool, CatBoostClassifier

RANDOM_STATE = 1
TEST_SIZE = 0.3
NEW_CLIENTS_SIZE = 0.5
BASIC_TRESHOLD = 0.6
N_SPLITS = 5

REPORT_FILE_PATH = '../reports/report.csv'


def data_split(df, create_new_clients=False, new_clients_size=NEW_CLIENTS_SIZE):
    if create_new_clients:
        all_clients = df['Наименование ДП'].unique()
        new_ids = all_clients.copy()
        np.random.seed(RANDOM_STATE)
        np.random.shuffle(new_ids)
        new_ids = new_ids[:int(len(all_clients) * new_clients_size)]
        if '2019' in df.year.unique():
            train = df[df.year.isin(['2019', '2020'])]
            test = df[df.year == '2021']
        else:
            train = df[df.year == '2020']
            test = df[df.year == '2021']
        train = train[~train['Наименование ДП'].isin(new_ids)]
    else:
        if '2019' in df.year.unique():
            train = df[df.year.isin(['2019', '2020'])]
            test = df[df.year == '2021']
        else:
            train = df[df.year == '2020']
            test = df[df.year == '2021']
    
    y_train = train.binary_target.astype(int)
    y_test = test.binary_target.astype(int)
    train = train.drop(columns=['year', 'Наименование ДП', 'binary_target'])
    test = test.drop(columns=['year', 'Наименование ДП', 'binary_target'])
    return train, test, y_train, y_test


def fit_predict(model, X_train, y_train, X_test, y_test, treshold=BASIC_TRESHOLD, plot_roc_auc=False):
    print(f'Fitting model {model} with treshold = {round(treshold, 2)}...')
    if str(model.__class__()).split('.')[-1].split()[0] == 'CatBoostClassifier':
        X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
        eval_set = Pool(X_val, y_val)
        model.fit(X_train_, y_train_, eval_set=eval_set)
    else:
        model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)[:, 1]
    preds = probas.copy()
    preds[np.where(preds < treshold)] = 0
    preds[np.where(preds >= treshold)] = 1
    if plot_roc_auc:
        disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.show()
    return model, preds, probas


def make_scores(y_test, preds, probas=None, use_probas=True):
    # print('Validate predictions...')
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    if use_probas:
        roc_auc = roc_auc_score(y_test, probas)
    else:
        roc_auc = 0
    return f1, precision, recall, acc, roc_auc


def validate_treshold(model, X, create_new_clients=False):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = data_split(X, create_new_clients=create_new_clients)
    treshold_list = np.arange(0.1, 1, 0.005)
    f1_list, precision_list, recall_list, acc_list = [], [], [], []
    model, preds, probas = fit_predict(model, X_train, y_train, X_test, y_test, treshold=BASIC_TRESHOLD)
    for treshold in treshold_list:
        preds = probas.copy()
        preds[np.where(preds < treshold)] = 0
        preds[np.where(preds >= treshold)] = 1
        f1, precision, recall, acc, roc_auc = make_scores(y_test, preds, probas=probas)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        acc_list.append(acc)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(treshold_list, f1_list, 'r', label='f1')
    ax.plot(treshold_list, precision_list, 'b', label='precision')
    ax.plot(treshold_list, recall_list, 'g', label='recall')
    ax.plot(treshold_list, acc_list, 'k', label='accuracy')
    ax.set_xlabel('treshold')
    ax.set_ylabel('Score')
    ax.legend()
    plt.show()
    

def make_report(model, X, treshold=BASIC_TRESHOLD, use_cross_val=False, create_new_clients=False, to_file=True, file_path=REPORT_FILE_PATH, comment=''):
    if use_cross_val:
        raise NotImplementedError()
        skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
        f1_list, precision_list, recall_list, acc_list, roc_list = [], [], [], [], []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model, preds, probas = fit_predict(model, X_train, y_train, X_test, y_test, treshold=treshold, plot_roc_auc=False)
            f1, precision, recall, acc, roc_auc = make_scores(y_test, preds, probas=probas)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            acc_list.append(acc)
            roc_list.append(roc_auc)
            
        f1 = np.mean(f1_list)
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        acc = np.mean(acc_list)
        roc_auc = np.mean(roc_list)
        
        f1_std = np.std(f1_list)
        precision_std = np.std(precision_list)
        recall_std = np.std(recall_list)
        acc_std = np.std(acc_list)
        roc_auc_std = np.std(roc_list)
    else:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        X_train, X_test, y_train, y_test = data_split(X, create_new_clients=create_new_clients)
        model, preds, probas = fit_predict(model, X_train, y_train, X_test, y_test, treshold=treshold, plot_roc_auc=True)
        f1, precision, recall, acc, roc_auc = make_scores(y_test, preds, probas=probas)
        f1_std, precision_std, recall_std, acc_std, roc_auc_std = 0, 0, 0, 0, 0
        
        #feature_importance
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            #shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
            shap.summary_plot(shap_values, X_test)
            plt.show()
        except:
            pass
        
    print('\033[92m' + f'F1 = {round(f1, 4)}, Precision = {round(precision, 4)}, Recall = {round(recall, 4)}, Accuracy = {round(acc, 4)}, ROC_AUC = {round(roc_auc, 4)}' + '\033[0m')
    if to_file:
        res = pd.DataFrame([[str(model.__class__()), model.get_params(), comment, round(treshold, 2), round(roc_auc, 4),
                             round(f1, 4), round(precision, 4), round(recall, 4), round(acc, 4), use_cross_val, 
                             round(roc_auc_std, 4), round(f1_std, 4), round(precision_std, 4), round(recall_std, 4), round(acc_std, 4)]], 
                           columns=['model', 'params', 'comment', 'treshold', 'roc_auc', 'f1', 'precision', 'recall', 'acc', 'use_cross_val', 
                                    'roc_auc_std', 'f1_std', 'precision_std', 'recall_std', 'acc_std'])
        if os.path.exists(file_path):
            res.to_csv(file_path, mode='a', header=False, index=False)
        else:
            res.to_csv(file_path, index=False)
