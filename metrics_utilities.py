import numpy as np
import time
from sklearn.metrics import precision_score, roc_curve, average_precision_score, balanced_accuracy_score, \
    accuracy_score, auc
from constants import WEIGHTED_METRICS, RESULT_CSV_PATH, RESULTS_CSV_PATH, MODELS_LIST, EVAL_FOLDS, OUR_MODEL
from sklearn.preprocessing import LabelBinarizer
import csv


def nanaverage(A, weights, axis):
    return np.nansum(A * weights, axis=axis) / ((~np.isnan(A)) * weights).sum(axis=axis)


def get_time_metrics(cv, X_train, y_train, X_test):
    t0 = time.time()
    cv.fit(X_train, y_train)
    training_time = time.time() - t0

    t0 = time.time()
    cv.predict(X_test)  # N x L, values in [0, 1]
    inference_time_per_entry = (time.time() - t0) / len(X_test)
    inference_time_for_1000 = inference_time_per_entry * 1000

    return training_time, inference_time_for_1000


def get_multiclass_metrics(y_test, y_test_pred):
    acc = balanced_accuracy_score(y_test, y_test_pred) if WEIGHTED_METRICS \
        else accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, average='weighted' if WEIGHTED_METRICS \
        else 'macro')

    return acc, prec


def binarize_zero_one(y, train_labels):
    y_ret = y.copy()
    y_ret[y == train_labels[0]] = 0
    y_ret[y == train_labels[1]] = 1
    return y_ret.astype(int)


def get_binary_metrics(y_test, y_test_pred, y_test_pred_per_label_probs, train_labels):
    per_label_fpr = []
    per_label_tpr = []
    per_label_pr_auc = []
    per_label_roc_auc = []

    def set_metrics(y_true, y_pred, y_pred_probs):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        per_label_fpr.append(fpr[np.asarray(thresholds) == 1][0])
        per_label_tpr.append(tpr[np.asarray(thresholds) == 1][0])
        per_label_roc_auc.append(auc(fpr, tpr))
        pr_auc = average_precision_score(y_true, y_pred_probs)
        per_label_pr_auc.append(pr_auc)

    test_labels = y_test.unique()
    test_labels_dist = (y_test.value_counts() / len(y_test)).values
    assert set(test_labels).issubset(set(train_labels)), "TEST LABELS NOT IN TRAIN LABELS"  # Shouldn't happen

    is_binary = len(train_labels) == 2

    if is_binary:  # Binary case is considered as single labeled
        curr_y_true = binarize_zero_one(y_test.values, train_labels)
        curr_y_pred = binarize_zero_one(y_test_pred, train_labels)
        curr_y_pred_probs = y_test_pred_per_label_probs[:, 1]

        set_metrics(curr_y_true, curr_y_pred, curr_y_pred_probs)

        weights = [1] if not WEIGHTED_METRICS else [np.mean(curr_y_true)]

    else:
        lb = LabelBinarizer()
        y_true_binarized = lb.fit_transform(y_test)
        y_pred_binarized = lb.transform(y_test_pred)
        lb_classes = list(lb.classes_)

        if len(lb_classes) == 2:
            y_true_binarized = np.hstack((y_true_binarized, 1-y_true_binarized))
            y_pred_binarized = np.hstack((y_pred_binarized, 1-y_pred_binarized))

        for label_idx, curr_label in enumerate(test_labels):
            curr_y_true = y_true_binarized[:, lb_classes.index(curr_label)]
            curr_y_pred = y_pred_binarized[:, lb_classes.index(curr_label)]
            curr_y_pred_probs = y_test_pred_per_label_probs[:, label_idx]

            set_metrics(curr_y_true, curr_y_pred, curr_y_pred_probs)

        weights = [1 for i in range(len(test_labels))] if not WEIGHTED_METRICS else test_labels_dist

    fpr = np.average(per_label_fpr, weights=weights)
    tpr = np.average(per_label_tpr, weights=weights)
    pr_auc = np.average(per_label_pr_auc, weights=weights)
    roc_auc = np.average(per_label_roc_auc, weights=weights)

    return fpr, tpr, pr_auc, roc_auc


def write_all_results(dbs_results, part_number=0):
    if part_number > 0:
        results_path = RESULTS_CSV_PATH + (".%d" % part_number)
    else:
        results_path = RESULTS_CSV_PATH
    with open(results_path, "w") as csvfile:

        fieldnames = ['dataset_name', 'alg_name']

        # getting the metrics name (using the first db and our model)
        first_db_name = list(dbs_results.keys())[0]
        metrics_names = list(dbs_results[first_db_name][OUR_MODEL][0].keys())
        fieldnames += metrics_names

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for db_idx, db_name in enumerate(dbs_results):
            for model_idx, model_name in enumerate(MODELS_LIST):
                for fold_num in range(EVAL_FOLDS):
                    info_dict = {'dataset_name': db_name, 'alg_name': model_name}

                    metrics_dict = dbs_results[db_name][model_name][fold_num]

                    writer.writerow({**info_dict, **metrics_dict})


def write_single_db_results(db_results, db_name):
    print("writing single db results %s" % db_name)
    with open(RESULT_CSV_PATH(db_name), "w") as csvfile:

        fieldnames = ['dataset_name', 'alg_name']

        # getting the metrics name (using the first db and our model)
        first_db_name = list(db_results.keys())[0]
        metrics_names = list(db_results[first_db_name][OUR_MODEL][0].keys())
        fieldnames += metrics_names

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_idx, model_name in enumerate(MODELS_LIST):
            for fold_num in range(EVAL_FOLDS):
                info_dict = {'dataset_name': db_name, 'alg_name': model_name}

                metrics_dict = db_results[model_name][fold_num]

                writer.writerow({**info_dict, **metrics_dict})