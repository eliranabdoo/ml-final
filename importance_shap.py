import matplotlib.pyplot as plt
import os
import pandas as pd
import shap
from constants import PLOTS_DIR, PLOTS_FORMAT, IMPORTANCE_TYPES, IMPORTANCE_DIR, PLOT_TOP_FEATURES, SHAP_DIR


def close_and_increase_plt():
    plt.close(globals()["figure_num"])
    globals()["figure_num"] += 1


def get_plot_path(innder_dir, desc):
    return os.path.join(PLOTS_DIR + '/' + innder_dir + '/'
                        + desc
                        + '.' + PLOTS_FORMAT)


def generate_importance(model, test_db_name):
    for imp_type in IMPORTANCE_TYPES:
        feature_important = model.get_booster().get_score(importance_type=imp_type)
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        ax = data.plot(kind='barh')
        imp_plot_path = get_plot_path(IMPORTANCE_DIR, '{test_db_name}_{imp_type}'.format(test_db_name=test_db_name, \
                                                                                         imp_type=imp_type))
        ax.figure.savefig(imp_plot_path, bbox_inches='tight')


def generate_shap(model, test_db_name, X_test):
    # For XGB>1.0.0
    # booster_bytearray = model.get_booster().save_raw()[4:]
    #
    # def myfun(self=None):
    #     return booster_bytearray
    #
    # model.get_booster().save_raw = myfun

    # create a SHAP explainer and values
    explainer = shap.TreeExplainer(model)

    # summary_plot
    plt.figure(globals()["figure_num"])
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, max_display=PLOT_TOP_FEATURES, show=False, plot_type="bar")

    shap_summary_plot_path = get_plot_path(SHAP_DIR, '{test_db_name}_shap'.format(test_db_name=test_db_name))
    plt.savefig(shap_summary_plot_path, bbox_inches='tight', format=PLOTS_FORMAT)
    close_and_increase_plt()
