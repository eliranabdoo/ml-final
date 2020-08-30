import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
# os.system("python -m pip install xgboost==1.0.0")
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.multiclass import OneVsRestClassifier
from datetime import datetime
from scipy.stats import friedmanchisquare
# os.system("python -m pip install scikit_posthocs")
from scikit_posthocs import posthoc_nemenyi_friedman
import dill
import os

from constants import *
from rboost import RBoost
from elpboost import ELPBoost
from importance_shap import *
from preprocessing import DelayedColumnTransformer
from metrics_utilities import *


def db_encode(db_name, X, y):
    """Performs fixed replacements of values to handle db-specific value anomalies
    """
    if y.dtype == np.object:
        y = y.replace(y_values_encoding)
    if db_name in X_nan_values:
        X = X.replace(X_nan_values[db_name], np.nan)
    if db_name in X_func_replacement:
        X = X.apply(X_func_replacement[db_name])
    return X, y


def summarize_metrics(folds_metrics):
    return np.average([folds_metrics[i][STAT_CHOSEN_METRIC] for i in range(len(folds_metrics))])


def main():
    PART_NUMBER = 0
    dataset_paths = [os.path.join(CLASS_DBS_PATH, dataset_name) for dataset_name in sorted(os.listdir(CLASS_DBS_PATH))]
    # [("db_name", read_cvs)]
    raw_dbs = [(os.path.basename(dataset_path), pd.read_csv(dataset_path)) for dataset_path in dataset_paths]
    # [("db_name", X, y)]
    raw_dbs = [(raw_db[0], \
                raw_db[1].loc[:, raw_db[1].columns != raw_db[1].columns[-1]], \
                raw_db[1].loc[:, raw_db[1].columns[-1]]) \
               for raw_db in raw_dbs]

    raw_dbs = sorted(raw_dbs, key=lambda x: len(x[1]))  # sort by db length

    if len(sys.argv) > 1:  # For distributed training of multiple dbs over multiple servers
        num_parts = int(sys.argv[1])
        curr_part = int(sys.argv[2])
        assert curr_part <= num_parts
        assert curr_part >= 1

        PART_NUMBER = curr_part

        print("working on dbs %s" % str(list(range(curr_part - 1, len(raw_dbs), num_parts))))
        raw_dbs = [raw_dbs[i] for i in range(curr_part - 1, len(raw_dbs), num_parts)]

    preprocessing = DelayedColumnTransformer([
        (np.object, [SimpleImputer(strategy='constant'), OneHotEncoder(handle_unknown='ignore')]),
        (np.number, [SimpleImputer(strategy='mean'), VarianceThreshold(0.0)])
    ])

    eval_metric = balanced_accuracy_score

    kf = StratifiedKFold(n_splits=EVAL_FOLDS, random_state=RANDOM_SEED)
    model = Pipeline(steps=[('model', RBoost() if USE_RBOOST else ELPBoost())])
    comp_model = Pipeline(steps=[('model', lgb.LGBMClassifier())])
    ova_model = OneVsRestClassifier(model)
    ova_comp_model = OneVsRestClassifier(comp_model)

    # {db_name: {our_model: reulsts, compare_model: results}}
    dbs_results = {}

    with open(os.path.join(WORKING_DIR, "bad-dbs.txt"), "w") as f:
        pass

    os.system('mkdir -p {}'.format(MODELS_DIR))

    for db_name, X, y in raw_dbs:
        dbs_results[db_name] = {}
        X, y = db_encode(db_name, X, y)

        N = len(X) * (1 - (1 / EVAL_FOLDS))

        fold_num = 1

        # list of results per fold
        folds_results = []
        comp_folds_results = []

        is_binary = len(y.unique()) == 2  # No special case for binary
        try:
            for train_index, test_index in kf.split(X, y):
                print("{}:{}:Fold_{}".format(datetime.now(), db_name, fold_num))
                # --- get fold and preprocess --- #
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                invalid_labels = set(y_test.unique()) - set(y_train.unique())
                # Introduce new labels, should occur only for outliers due to StratifiedKFold
                # Main assumption in classification is that all labels are known upfront
                if len(invalid_labels) > 0:
                    X_train = pd.concat([X_train, pd.DataFrame(
                        [[np.nan for _ in range(len(X_train.columns))] for _ in range(len(invalid_labels))],
                        columns=X_train.columns)], ignore_index=True)
                    y_train = y_train.append(pd.Series(list(invalid_labels)), ignore_index=True)
                    X_train, y_train = db_encode(db_name, X_train, y_train)

                preprocessing.fit(X_train, y_train)
                X_train = preprocessing.transform(X_train)
                X_test = preprocessing.transform(X_test)

                model_params = {
                    'estimator__model__kappa': [1 / 3, 1 / N, 2 / N, 3 / N],
                    'estimator__model__T': [3, 5, 10],
                    'estimator__model__reg': [1, 10, 20, 50, 100],
                    'estimator__model__silent': [True],
                    'estimator__model__verbose': [False]
                }

                # --- random search --- #
                cv = RandomizedSearchCV(estimator=ova_model, param_distributions=model_params,
                                        scoring=make_scorer(eval_metric), cv=HPT_FOLDS,
                                        n_iter=RANDOM_CV_ITER, random_state=RANDOM_SEED)
                comp_cv = RandomizedSearchCV(estimator=ova_comp_model, param_distributions=comp_model_params,
                                             scoring=make_scorer(eval_metric), cv=HPT_FOLDS,
                                             n_iter=RANDOM_CV_ITER, random_state=RANDOM_SEED)

                curr_fold_results = {'fold_num': fold_num}
                curr_fold_comp_results = {'fold_num': fold_num}

                # --- measure times - FIT + INFER --- #
                print("Training our model")
                curr_fold_results['train_time'], curr_fold_results['infer_time'] = get_time_metrics(cv, X_train,
                                                                                                    y_train,
                                                                                                    X_test)
                print("Finished training our model")
                print("Training comparison model")
                curr_fold_comp_results['train_time'], curr_fold_comp_results['infer_time'] = get_time_metrics(comp_cv,
                                                                                                              X_train,
                                                                                                              y_train,
                                                                                                              X_test)
                print("Finished training comparison model")

                # --- save trained models --- #
                model_path = MODELS_DIR + "/model_fold_" + str(fold_num) + "_db_name_" + db_name
                comp_model_path = MODELS_DIR + "/comp_model_fold_" + str(fold_num) + "_db_name_" + db_name
                dill.dump(cv.best_estimator_, open(model_path, 'wb'))
                dill.dump(comp_cv.best_estimator_, open(comp_model_path, 'wb'))

                # --- register best params --- #
                best_comp_params = comp_cv.best_params_
                best_params = cv.best_params_
                curr_fold_comp_results['best_params'] = best_comp_params
                curr_fold_results['best_params'] = best_params
                # --- get predictions for MultiRBoost --- #
                y_test_pred_per_label_scores = cv.predict_proba(X_test)
                y_test_pred = cv.predict(X_test)

                train_labels = cv.best_estimator_.classes_
                comp_train_labels = comp_cv.best_estimator_.classes_  # can be sorted differently

                # --- get predictions for LightGBM --- #
                y_test_pred_comp_per_label_scores = comp_cv.predict_proba(X_test)
                y_test_pred_comp = comp_cv.predict(X_test)

                # --- replace nans with uniform - fixes an error in OneVsRest --- #
                y_test_pred_comp_per_label_scores[np.isnan(y_test_pred_comp_per_label_scores)] = 1.0 / \
                                                                                                 y_test_pred_comp_per_label_scores.shape[
                                                                                                     1]

                y_test_pred_per_label_scores[np.isnan(y_test_pred_per_label_scores)] = 1.0 / \
                                                                                       y_test_pred_per_label_scores.shape[
                                                                                           1]

                # metrics applicable in multiclass setting ---accuracy, precision--- #
                multiclass_metrics_dict = {0: 'accuracy', 1: 'precision'}

                multiclass_metrics = get_multiclass_metrics(y_test, y_test_pred)
                multiclass_comp_metrics = get_multiclass_metrics(y_test, y_test_pred_comp)

                for metric_pos, metric_name in multiclass_metrics_dict.items():
                    curr_fold_results[metric_name] = multiclass_metrics[metric_pos]
                    curr_fold_comp_results[metric_name] = multiclass_comp_metrics[metric_pos]

                # Metrics only applicable in a binary setting ---fpr, tpr, pr_auc, roc-auc--- #
                binary_metrics_dict = {0: 'fpr', 1: 'tpr', 2: 'pr_auc', 3: 'roc_auc'}

                binary_metrics = get_binary_metrics(y_test, y_test_pred, y_test_pred_per_label_scores, \
                                                    train_labels)
                binary_comp_metrics = get_binary_metrics(y_test, y_test_pred_comp, y_test_pred_comp_per_label_scores, \
                                                         comp_train_labels)

                for metric_pos, metric_name in binary_metrics_dict.items():
                    curr_fold_results[metric_name] = binary_metrics[metric_pos]
                    curr_fold_comp_results[metric_name] = binary_comp_metrics[metric_pos]

                # add the current fold results to the results list
                folds_results.append(curr_fold_results)
                comp_folds_results.append(curr_fold_comp_results)

                fold_num += 1
            dbs_results[db_name][OUR_MODEL] = folds_results
            dbs_results[db_name][COMP_MODEL] = comp_folds_results

            write_single_db_results(dbs_results[db_name], db_name)

        except Exception as e:
            print("ERROR!", e)
            # catching weird values
            with open(os.path.join(WORKING_DIR, "bad-dbs.txt"), "a") as f:
                dbs_results.pop(db_name)
                f.write("{db_name}: {error}\n".format(db_name=db_name, error=e))

            continue

    print(dbs_results)
    write_all_results(dbs_results, PART_NUMBER)

    print("Done writing results in part %d" % PART_NUMBER)

    # --- Statistical Tests Section --- #
    # --- Friedman Test --- #
    models_measures = np.zeros(shape=(len(dbs_results), 2))

    for model_idx, model_name in enumerate(MODELS_LIST):
        for db_idx, db_name in enumerate(dbs_results):
            models_measures[db_idx][model_idx] = np.average([dbs_results[db_name][model_name][i][STAT_CHOSEN_METRIC] \
                                                             for i in range(EVAL_FOLDS)])

    stats_per_db = [models_measures[i, :] for i in range(models_measures.shape[0])]

    p_value = friedmanchisquare(*stats_per_db).pvalue
    print(p_value)

    if p_value <= P_THRESH:
        print("Statistically significant!")
        post_hoc_res = posthoc_nemenyi_friedman(models_measures)
        print("nemenyi post-hoc result: {res}".format(res=post_hoc_res))

    else:
        print("Not statistically significant!")

    # --- Meta Learning Section --- #
    per_dataset_winner = {}
    for db_name in dbs_results:
        our_model_metrics = dbs_results[db_name][OUR_MODEL]
        comp_model_metrics = dbs_results[db_name][COMP_MODEL]
        we_win = summarize_metrics(our_model_metrics) >= summarize_metrics(comp_model_metrics)
        per_dataset_winner[db_name.split('.')[0]] = 1 if we_win else -1

    X_raw = pd.read_csv(META_DBS_PATH, header=0, index_col='dataset')
    X = X_raw.loc[list(per_dataset_winner.keys()), :]
    y = pd.Series([per_dataset_winner[db_name] for db_name in per_dataset_winner])

    db_names = [db_name for db_name in per_dataset_winner]

    loo = LeaveOneOut()
    meta_model_results = {}

    os.system('mkdir -p {plots_dir}/{inner_dir}'.format(plots_dir=PLOTS_DIR, inner_dir=IMPORTANCE_DIR))
    os.system('mkdir -p {plots_dir}/{inner_dir}'.format(plots_dir=PLOTS_DIR, inner_dir=SHAP_DIR))

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        curr_dataset = X_test.index[0]
        meta_model = xgb.XGBClassifier(booster='gbtree')
        meta_model.fit(X_train, y_train)
        y_pred = meta_model.predict(X_test)[0]

        meta_model_results[curr_dataset] = y_pred

        generate_importance(meta_model, db_names[test_index[0]])
        generate_shap(meta_model, db_names[test_index[0]], X_test)

    y_pred = pd.Series([meta_model_results[db_name] for db_name in meta_model_results])

    print("Meta Model Accuracy: %f" % accuracy_score(y, y_pred))


if __name__ == "__main__":
    main()
