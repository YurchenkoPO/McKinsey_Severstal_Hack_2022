import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import shap

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay, accuracy_score, roc_curve, log_loss
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from catboost import Pool, CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

TARGET_COL = 'binary_target'
ALL_TARGET_COLS = ['binary_target', 'target_more30days', 'target_more90days']

RANDOM_STATE = 1
TEST_SIZE = 0.3
NEW_CLIENTS_SIZE = 0.4
BASIC_THRESHOLD = 0.6
N_SPLITS = 5

REPORT_FILE_PATH = '../reports/report.csv'


def data_split(df, cols2drop=[], target_col=TARGET_COL, create_new_clients=False, new_clients_size=NEW_CLIENTS_SIZE, random_state=RANDOM_STATE):
    if create_new_clients:
        all_clients = df['Наименование ДП'].unique()
        new_ids = all_clients.copy()
        np.random.seed(random_state)
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
    
    y_train = train[target_col].astype(int)
    y_test = test[target_col].astype(int)
    train = train.drop(columns=['year', 'Наименование ДП'] + ALL_TARGET_COLS + cols2drop)
    test = test.drop(columns=['year', 'Наименование ДП'] + ALL_TARGET_COLS + cols2drop)
    return train, test, y_train, y_test

def calibrate_model(base_model, X_val, Y_val, calib_coeff):
    sample_weights = [1 if x == 0 else calib_coeff for x in Y_val.values]
    calib_model = CalibratedClassifierCV(base_estimator=base_model, method='sigmoid', cv=3)
    calib_model.fit(X_val, Y_val, sample_weight=sample_weights)

    return calib_model


def fit_predict(model, X_train, y_train, X_test, y_test, use_calib, calib_coeff, threshold=BASIC_THRESHOLD, plot_roc_auc=False):
    if 'CatBoostClassifier' in  str(model.__class__()):
        if model.get_params()['use_best_model']:
            X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            eval_set = Pool(X_val, y_val)
            model.fit(X_train_, y_train_, eval_set=eval_set)
            if use_calib:
                model = calibrate_model(model, X_train_, y_train_, calib_coeff)
                print('MODEL HAS BEEN CALIBRATED!')
        else:
            if use_calib:
                model = calibrate_model(model, X_train, y_train, calib_coeff)
                print('MODEL HAS BEEN CALIBRATED!')
            else:
                model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)[:, 1]
    preds = probas.copy()
    preds[np.where(preds < threshold)] = 0
    preds[np.where(preds >= threshold)] = 1
    
    train_probas = model.predict_proba(X_train)[:, 1]
    train_preds = train_probas.copy()
    train_preds[np.where(train_preds < threshold)] = 0
    train_preds[np.where(train_preds >= threshold)] = 1
    if plot_roc_auc:
        fig, ax = plt.subplots()
        disp = RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name='test')
        disp = RocCurveDisplay.from_estimator(model, X_train, y_train, ax=ax, name='train')
        plt.show()
    return model, preds, probas, train_preds, train_probas


def make_scores(y_test, preds, probas=None, use_probas=True):
    # print('Validate predictions...')
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    if use_probas is True and probas is not None:
        roc_auc = roc_auc_score(y_test, probas)
    else:
        roc_auc = 0
    return f1, precision, recall, acc, roc_auc


def validate_threshold(model, X, target_col=TARGET_COL, create_new_clients=False):
    print(f'Target = {target_col}')
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = data_split(X, target_col=target_col, create_new_clients=create_new_clients)
    threshold_list = np.arange(0.1, 1, 0.005)
    f1_list, precision_list, recall_list, acc_list = [], [], [], []
    model, preds, probas, _, _ = fit_predict(model, X_train, y_train, X_test, y_test, threshold=BASIC_THRESHOLD)
    for threshold in threshold_list:
        preds = probas.copy()
        preds[np.where(preds < threshold)] = 0
        preds[np.where(preds >= threshold)] = 1
        f1, precision, recall, acc, roc_auc = make_scores(y_test, preds, probas=probas)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        acc_list.append(acc)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(threshold_list, f1_list, 'r', label='f1')
    ax.plot(threshold_list, precision_list, 'b', label='precision')
    ax.plot(threshold_list, recall_list, 'g', label='recall')
    ax.plot(threshold_list, acc_list, 'k', label='accuracy')
    ax.set_xlabel('threshold')
    ax.set_ylabel('Score')
    ax.legend()
    plt.show()
    

def make_report(model, X, cols2drop, use_calib, calib_coeff, target_col=TARGET_COL, threshold=BASIC_THRESHOLD, use_cross_val=False, create_new_clients=False,
                to_file=True, file_path=REPORT_FILE_PATH, comment='', need_val=False, to_plot=True, random_state=RANDOM_STATE,
                suppress_prints=False):
    if not suppress_prints:
        print(f'Fitting model {model} with threshold = {round(threshold, 5)}...')
        
    if use_cross_val:
        raise NotImplementedError('No need because test data is always 2021 and we can`t use it as train data')
        skf = StratifiedKFold(n_splits=N_SPLITS, random_state=random_state, shuffle=True)
        f1_list, precision_list, recall_list, acc_list, roc_list = [], [], [], [], []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model, preds, probas, _, _ = fit_predict(model, X_train, y_train, X_test, y_test, threshold=threshold, plot_roc_auc=False, use_calib=use_calib)
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
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)
        X_train, X_test, y_train, y_test = data_split(X, cols2drop=cols2drop, target_col=target_col, create_new_clients=create_new_clients)
        if need_val:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=random_state, stratify=y_train)
        model, preds, probas, train_preds, train_probas = fit_predict(model, X_train, y_train, X_test, y_test, calib_coeff=calib_coeff, threshold=threshold, use_calib=use_calib, plot_roc_auc=to_plot)
        f1, precision, recall, acc, roc_auc = make_scores(y_test, preds, probas=probas)
        train_f1, train_precision, train_recall, train_acc, train_roc_auc = make_scores(y_train, train_preds, probas=train_probas)
        f1_std, precision_std, recall_std, acc_std, roc_auc_std = 0, 0, 0, 0, 0
        
        #feature_importance
        if to_plot:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                #shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
                shap.summary_plot(shap_values, X_test)
                plt.show()
            except:
                pass
            
    
    if not suppress_prints:
        print(f'\nTRAIN: F1 = {round(train_f1, 3)}, Precision = {round(train_precision, 3)}, Recall = {round(train_recall, 3)}, Accuracy = {round(train_acc, 3)}, ROC_AUC = {round(train_roc_auc, 3)}')
        print('\033[92m' + f'TEST: F1 = {round(f1, 3)}, Precision = {round(precision, 3)}, Recall = {round(recall, 3)}, Accuracy = {round(acc, 3)}, ROC_AUC = {round(roc_auc, 3)}' + '\033[0m' + '\n')
    
    if to_file:
        res = pd.DataFrame([[str(model.__class__()), model.get_params(), comment, round(threshold, 6), round(roc_auc, 3),
                             round(f1, 3), round(precision, 3), round(recall, 3), round(acc, 3), use_cross_val, 
                             round(train_roc_auc, 3), round(train_f1, 3), round(train_precision, 3), round(train_recall, 3), round(train_acc, 3)]], 
                           columns=['model', 'params', 'comment', 'threshold', 'roc_auc', 'f1', 'precision', 'recall', 'acc', 'use_cross_val', 
                                    'roc_auc_train', 'f1_train', 'precision_train', 'recall_train', 'acc_train'])
        if os.path.exists(file_path):
            res.to_csv(file_path, mode='a', header=False, index=False)
        else:
            res.to_csv(file_path, index=False)
    return roc_auc, f1, precision, recall, acc, model
            
            
def make_report_with_best_threshold(model, df, calib_coeff=0, cols2drop=[], create_new_clients=False,
                                    to_file=True, file_path=REPORT_FILE_PATH, comment='', 
                                    target_col=TARGET_COL, num_random_states=1, use_calib=False):
    print(f'Target = {target_col}')
    threshold_list = []
    for rs in np.arange(1, 10 * num_random_states + 1, 10).astype(int):
        _, _, _, _, _, model = make_report(model, df, calib_coeff=calib_coeff, cols2drop=cols2drop, target_col=target_col, threshold=0.5, to_file=False,
                        create_new_clients=create_new_clients, need_val=True,
                        to_plot=False, random_state=rs, suppress_prints=True, use_calib=use_calib)

        X_train, X_test, y_train, y_test = data_split(df, cols2drop=cols2drop, target_col=target_col, 
                                                      create_new_clients=create_new_clients, random_state=rs)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=rs, stratify=y_train)
        probas = model.predict_proba(X_val)[:, 1]

        fpr, tpr, threshold = roc_curve(y_val, probas)
        i = np.arange(len(tpr))
        roc = pd.DataFrame(
            {
                "tf": pd.Series(tpr - (1 - fpr), index=i),
                "tpr": pd.Series(tpr, index=i),
                "fpr": pd.Series(fpr, index=i),
                "threshold": pd.Series(threshold, index=i),
            }
        )
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
        threshold_list.append(np.mean(list(roc_t["threshold"])))

    roc_auc, f1, precision, recall, acc, model = make_report(model , df, calib_coeff=calib_coeff, use_calib=use_calib, cols2drop=cols2drop, target_col=target_col, threshold=np.mean(threshold_list),
                                                     to_file=to_file, file_path=file_path,
                                                     comment=comment, create_new_clients=create_new_clients, need_val=False)
    if use_calib:
        return probas

def hyperopt_for_catboost(X, target_col=TARGET_COL):
    print(f'Target = {target_col}')
    
    X_train, X_test, y_train, y_test = data_split(X, target_col=target_col)

    def get_catboost_params(space):
        params = dict()
        params['learning_rate'] = space['learning_rate']
        # params['class_w'] = space['class_w']
        params['depth'] = int(space['depth'])
        params['l2_leaf_reg'] = space['l2_leaf_reg']
        # params['iterations'] = int(space['iterations'])
        return params
    
    global obj_call_count, cur_best_score, cur_best_loss
    obj_call_count = 0
    cur_best_loss = np.inf
    cur_best_score = 0

    def objective(space):
        global obj_call_count, cur_best_score, cur_best_loss
        obj_call_count += 1
        print('\nCatBoost objective call #{} cur_best_score={:7.5f}'.format(obj_call_count, cur_best_score) )
        params = get_catboost_params(space)

        sorted_params = sorted(space.items(), key=lambda z: z[0])
        params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
        print('Params: {}'.format(params_str) )

        model = CatBoostClassifier(iterations=500, #params['iterations'],
                                   depth=params['depth'], #5
                                   l2_leaf_reg=params['l2_leaf_reg'], 
                                   learning_rate=params['learning_rate'],
                                   loss_function='Logloss',
                                   use_best_model=False,
                                   eval_metric='AUC',
                                   verbose=False,
                                   # class_weights=[1, 0.01], # params['class_w']
                                   random_seed=RANDOM_STATE,
                                    )

        model, preds, probas, _, _ = fit_predict(model, X_train, y_train, X_test, y_test, threshold=0.5, plot_roc_auc=False)
        f1, precision, recall, acc, roc_auc = make_scores(y_test, preds, probas=probas)
        test_loss = log_loss(y_test, preds, labels=[0, 1])

        nb_trees = model.tree_count_
        print('nb_trees={}'.format(nb_trees))

        if test_loss < cur_best_loss:
            cur_best_loss = test_loss
            cur_best_score = roc_auc
            print('\033[92m' + 'NEW BEST LOSS={}'.format(cur_best_loss) + '\033[0m')
            print('\033[92m' + 'NEW BEST ROC_AUC={}'.format(roc_auc) + '\033[0m')
            
        return {'loss':test_loss, 'status': STATUS_OK }
    
    space = {
        'learning_rate': hp.loguniform('learning_rate', -6, -1),
        # 'class_w': hp.uniform('class_w', 1e-4, 1e-1),
        'depth': hp.quniform("depth", 2, 7, 1),
        # 'iterations': hp.quniform('iterations', 200, 5000, 1),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 3, 10),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        verbose=True
    )
    print('-'*50)
    print('The best params:')
    print(best)
    return best
