import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class GetUnique(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.groupby('customerid')[self.columns].unique().apply(
            lambda x: sum(~pd.isna(x))).to_frame()

class GetUid(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return pd.DataFrame(X[self.columns].unique())
    
class GetMean(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.groupby('customerid')[self.columns].mean().to_frame()
        
class DaydeltaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, t1_col, t2_col, col_name):
        self.t1_col = t1_col
        self.t2_col = t2_col
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xr = X.copy() 
        Xr['date1'] = pd.to_datetime(Xr[self.t1_col]) 
        Xr['date2'] = pd.to_datetime(Xr[self.t2_col])
        Xr[self.col_name] = Xr[['date1','date2']].apply(lambda x: (x[0] - x[1]).days*(x[0] > x[1]), axis=1)
        
        return  Xr.groupby('customerid')[self.col_name].mean().to_frame()

class TimedeltaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, t1_col, t2_col, col_name):
        self.t1_col = t1_col
        self.t2_col = t2_col
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xr = X.copy() 
        Xr['date1'] = pd.to_datetime(Xr[self.t1_col]) 
        Xr['date2'] = pd.to_datetime(Xr[self.t2_col])
        Xr[self.col_name] = Xr[['date1','date2']].apply(lambda x: (x[0] - x[1]).seconds/60, axis=1)
        
        return Xr.groupby('customerid')[self.col_name].mean().to_frame()

class AgeYears(BaseEstimator, TransformerMixin):
    def __init__(self, t1_col, t2_col, col_name):
        self.t1_col = t1_col
        self.t2_col = t2_col
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        Xr = X.copy() 
        Xr['date1'] = pd.to_datetime(Xr[self.t1_col]) 
        Xr['date2'] = pd.to_datetime(Xr[self.t2_col])
        Xr[self.col_name] = Xr[['date1','date2']].apply(lambda x: (x[0].year - x[1].year), axis=1)
        Xr.drop(columns=[self.t1_col,'date1','date2'], inplace=True)
        
        return  Xr

class ApprovalPeriod(BaseEstimator, TransformerMixin):
    def __init__(self, t1_col, t2_col, col_name):
        self.t1_col = t1_col
        self.t2_col = t2_col
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        Xr = X.copy() 
        Xr['date1'] = pd.to_datetime(Xr[self.t1_col]) 
        Xr['date2'] = pd.to_datetime(Xr[self.t2_col])
        Xr[self.col_name] = Xr[['date1','date2']].apply(lambda x: (x[0] - x[1]).seconds/60, axis=1)
        Xr.drop(columns=[self.t1_col, self.t2_col, 'date1', 'date2'], inplace=True)
        
        return  Xr
    
class ReferredTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, col_name):
        self.columns = columns
        self.col_name = col_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        Xr = X.copy() 
        Xr[self.col_name] = Xr[self.columns].apply(lambda x: (not pd.isna(x)) * 1)
        
        return Xr
        
class VarFillNa(BaseEstimator, TransformerMixin):
    def __init__(self, columns, var):
        self.columns = columns
        self.var = var
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        Xr = X.copy()
        Xr[self.columns].fillna(value=self.var, inplace=True)
        
        return Xr

class ColumnDropTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        Xr = X.copy()
        Xr.drop(columns=self.columns, inplace=True)
        
        return Xr

class StatFillNa(BaseEstimator, TransformerMixin):
    def __init__(self, col_1, col_2, stat):
        self.col_1 = col_1
        self.col_2 = col_2
        self.stat = stat
        
    def fit(self, X, y=None):
        return self
    
    def _impute(self, series, loanflag):
        res = []
        for i in range(len(series)):
            if pd.isnull(series)[i]:
                if loanflag[i] == 'Bad':
                    res.append(self.stat_bad)
                elif loanflag[i] == 'Good':
                    res.append(self.stat_good)
            else:
                res.append(series[i])
        return pd.Series(res)
    
    def transform(self, X):
        Xr = X.copy()
        if self.stat == 'Mean':
            self.stat_bad = Xr.groupby(self.col_2)[self.col_1].mean()[0]
            self.stat_good = Xr.groupby(self.col_2)[self.col_1].mean()[1]
        elif self.stat == 'Median':
            self.stat_bad = Xr.groupby(self.col_2)[self.col_1].median()[0]
            self.stat_good = Xr.groupby(self.col_2)[self.col_1].median()[1]
        elif self.stat == 'Mode':
            self.stat_bad = Xr.groupby(self.col_2)[self.col_1].value_counts().Bad.index[0]
            self.stat_good = Xr.groupby(self.col_2)[self.col_1].value_counts().Good.index[0]
        Xr[self.col_1] = self._impute(Xr[self.col_1].values, Xr[self.col_2].values)
        
        return Xr
    
class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        Xr = X.copy()
        return pd.get_dummies(Xr, columns=self.columns, drop_first=True)