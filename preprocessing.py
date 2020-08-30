from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DelayedColumnTransformer(TransformerMixin, BaseEstimator):
    """Wraps Column transformer in a pipeline, divided by dtypes
    Used to avoid a single, parallel, activation of the transform function, to use it in a sequential manner"""
    @staticmethod
    def get_dtype_columns_indices(df, dtype):
        columns = list(df.columns)
        return [columns.index(col) for col in df.select_dtypes(dtype).columns]

    def __init__(self, dtype_to_transformers):
        self.dtype_to_transformers = dtype_to_transformers
        self.pipeline = None

    def fit(self, X, y=None):
        self.pipeline = ColumnTransformer([
            ("%s" % str(dtype),
             Pipeline([("%s_%d" % (transformer.__class__.__name__, idx), transformer) for transformer in transformers]),
             self.get_dtype_columns_indices(X, dtype)) for
            idx, (dtype, transformers) in enumerate(self.dtype_to_transformers)]
            , remainder='drop')

        return self.pipeline.fit(X, y)

    def transform(self, X, y=None):
        res = self.pipeline.transform(X)
        return res
