import numpy as np
import pandas as pd
import json
from sklearn.model_selection import (train_test_split)
from sklearn.preprocessing import (LabelEncoder)

class Dataset:
    def __init__(self, file_path:str) -> None:
        # Importing and Processing the Dataframe
        self.df, self.cols, self.target_encoder = self._process_dataframe(file_path)

        # Finds the Numerical Columns inside the Dataset
        self.numerical_cols = self.df[self.cols[0:-1]].select_dtypes(include=np.number).columns.tolist()

        # Performs Binning over the Numerical Data
        self._perform_binning()

        # Gets the data and target of the dataset
        self.data, self.target = self._get_data_target()

        # Train and Test Dataframes used only for better visualization of the data
        self.Train_df, self.Test_df = None, None
        
    def _process_dataframe(self, file_path:str) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
        # Reading the .csv file
        df = pd.read_csv(file_path, keep_default_na=False, na_values=['NaN', 'nan'])

        # Encoding the Target
        label_encoder = LabelEncoder()
        target_column = df.columns[-1]
        df[target_column] = label_encoder.fit_transform(df[target_column])

        # Try to drop a potential columns 'ID' since it does not add anything to the predictive Model
        try:
            df.drop(columns=['ID'], inplace=True)
        except:
            pass
        
        return df, df.columns, label_encoder
    
    def _perform_binning(self) -> None:
        # Binning allows to convert numerical data into categorical variables (puts the values in grouped intervals)
        for numerical_column in self.numerical_cols:
            self.df[numerical_column] = pd.qcut(self.df[numerical_column], q=[0, .3, .7, 1])
            # self.df[numerical_column] = pd.qcut(self.df[numerical_column], q=[0, .25, .5, .75, 1])

    def _get_data_target(self) -> tuple[np.ndarray, np.ndarray]:
        # Filtering columns
        X_Cols = self.cols[0:-1]
        y_Col = self.cols[-1]

        # Partitioning the Data
        X = self.df[X_Cols].to_numpy()
        y = self.df[y_Col].squeeze().to_numpy()

        return X, y

    def _process_train_test_split(self, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray) -> None:
        # Convert the Train set into a DataFrame
        X_train_df = pd.DataFrame(X_train, columns=self.df.columns[0:-1])
        y_train_srs = pd.Series(self.target_encoder.inverse_transform(y_train), name=self.df.columns[-1])
        self.Train_df = pd.concat([X_train_df, y_train_srs], axis=1)
        
        # Convert the Test set into a DataFrame
        X_test_df = pd.DataFrame(X_test, columns=self.df.columns[0:-1])
        y_test_srs = pd.Series(self.target_encoder.inverse_transform(y_test), name=self.df.columns[-1])
        self.Test_df = pd.concat([X_test_df, y_test_srs], axis=1)
    
    def train_test_split(self, X=None, y=None, test_size=0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Setting Default Values
        X = self.data if X is None else X
        y = self.target if y is None else y

        # Splitting the Data
        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_size, random_state=13, shuffle=True, stratify=y)

        # Updating the Test and Train Dataframes
        self._process_train_test_split(X_train, X_test, y_train, y_test)
        
        return (X_train, X_test, y_train, y_test)

# Other Functions used to manage data

def calc_learning_curve_points(Model, X:np.ndarray, y:np.ndarray, n_itr=20, min_train_samples=1, n_points=10) -> list[tuple]:
    # Calculates a list of points to plot the learning curve
    points = []
    n_samples = len(X)
    max_test_size = (n_samples - min_train_samples) / n_samples
    test_sizes = np.linspace(min_train_samples / n_samples, max_test_size, n_points)
    
    for test_size in test_sizes:
        accuracies = []
        for _ in range(n_itr):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=13)
            model = Model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(sum(y_test ==  y_pred) / len(y_test))
        
        average_accuracy = np.mean(accuracies)
        train_set_size = n_samples - int(n_samples * test_size)
        points.append((train_set_size/n_samples*100, average_accuracy))
    
    return points

def Save_json_file(content: dict, file_path:str) -> None:
    # Saves a Dictionary as a json file
    with open(file_path, "w") as f:
        json.dump(content , f)
        
def Load_json_file(file_path:str) -> any:
    # Loads a json file - creates a dictionary
    with open(file_path) as f:
        return json.load(f)