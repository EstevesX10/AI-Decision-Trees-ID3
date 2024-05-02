import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split)
from sklearn.preprocessing import (LabelEncoder)

class Dataset:
    def __init__(self, file_path) -> None:
        self.df, self.cols, self.target_encoder = self._process_dataframe(file_path)
        self.numerical_cols = self.df[self.cols[0:-1]].select_dtypes(include=np.number).columns.tolist()
        
        self._perform_binning()
        self.data, self.target = self._get_data_target()
    
    def _process_dataframe(self, file_path):
        df = pd.read_csv(file_path, keep_default_na=False, na_values=['NaN', 'nan'])
        
        label_encoder = LabelEncoder()
        target_column = df.columns[-1]
        df[target_column] = label_encoder.fit_transform(df[target_column])

        try:
            df.drop(columns=['ID'], inplace=True)
        except:
            pass
        
        return df, df.columns, label_encoder
    
    def _perform_binning(self):
        # Binning allows to convert numerical data into categorical variables (puts the values in grouped intervals)
        for numerical_column in self.numerical_cols:
            self.df[numerical_column] = pd.qcut(self.df[numerical_column], q=[0, .3, .7, 1])
            # self.df[numerical_column] = pd.qcut(self.df[numerical_column], q=[0, .25, .5, .75, 1])

    def _get_data_target(self):
        X_Cols = self.cols[0:-1]
        y_Col = self.cols[-1]

        X = self.df[X_Cols].to_numpy()
        y = self.df[y_Col].squeeze().to_numpy()

        return X, y

    def train_test_split(self, X=None, y=None, test_size=0.3):
        X = self.data if X is None else X
        y = self.target if y is None else y
        return train_test_split(X, y, test_size=test_size, random_state=13, shuffle=True, stratify=y)