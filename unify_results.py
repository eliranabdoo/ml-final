import os
import csv
from collections import defaultdict
import pandas as pd

RESULTS_DIR = "./output/"
MODEL_KEY = 'alg_name'
FOLD_KEY = 'fold_num'
DB_NAME_KEY = 'dataset_name'

results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for file_path in os.listdir(RESULTS_DIR):
    if file_path.endswith(".csv"):
        df = pd.read_csv(os.path.join(RESULTS_DIR, file_path))
        for row in df.to_dict(orient='records'):
            model = row[MODEL_KEY]
            fold_num = row[FOLD_KEY]
            db_name = row[DB_NAME_KEY]
            row.pop(MODEL_KEY)
            row.pop(FOLD_KEY)
            row.pop(DB_NAME_KEY)
            results[db_name][model][fold_num] = row
print(results)
print(results['labor.csv'].keys())
