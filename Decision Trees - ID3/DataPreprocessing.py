import numpy as np
import pandas as pd
from ID3 import (DecisionTree)

class Dataset:
    def __init__(self, file_path):
        # Store the original dataframe
        self.df, self.encoded_df, self.y_decoder = self._process_dataframe(file_path)

        # Separates the data, target and encodes the target values returning 2 arrays and a dictionary
        self.data, self.target = self._get_data_target()
        
        # Variable that helps with printing the tree
        self.cols = self.encoded_df.columns

        # Getting the Numerical and Categorical columns within the Features (Essencial to Plot the Tree)
        self.num_cols = self.df[self.df.columns[0:-1]]._get_numeric_data().columns
        self.cat_cols = list(set(self.df.columns) - set(self.num_cols))

    def _label_encoder(self, array):
        # Find Unique Values
        unique_labels = sorted(np.unique(array))
        
        # Generate a mapping from label to integer
        label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Creating a Label Decoder
        label_decoder = {idx:label for label, idx in label_encoder.items()}
        
        return label_encoder, label_decoder
    
    def _process_dataframe(self, file_path):
        # Reading .csv file
        df = pd.read_csv(file_path)

        # Removing ID Column if existent
        if (df.columns[0] == 'ID'):
            df.drop(columns=['ID'], inplace=True)

        # Separating Features and Labels
        X = df[df.columns[0:-1]]
        y = df[df.columns[-1]]

        cols = X.columns
        num_cols = X._get_numeric_data().columns

        # Getting the Categorical columns within the Features
        cat_cols = list(set(cols) - set(num_cols))
        
        # Hot Encoding the categoricaL Features
        new_X = pd.get_dummies(X, prefix=cat_cols, dtype='int')

        # Concatenating the encoded features with the labels
        encoded_df = pd.concat([new_X, y], axis=1)

        # Sort based on the labels column
        encoded_df.sort_values(by=[encoded_df.columns[-1]], inplace=True)

        # Encode Labels
        labels_col = encoded_df.columns[-1]
        unique_labels = sorted(np.unique(encoded_df[labels_col]))

        label_encoder, label_decoder = self._label_encoder(encoded_df[encoded_df.columns[-1]].to_numpy())
        encoded_df.replace({labels_col: label_encoder}, inplace=True)
        
        return df, encoded_df, label_decoder

    def _get_data_target(self):
        # Defining the Target and Label Columns

        # X_Cols starts in 1 because we do not need the ID Column
        X_Cols = self.encoded_df.columns[0:-1]
        y_Col = self.encoded_df.columns[-1]
    
        # Splitting the Dataframe into features and label
        X = self.encoded_df[X_Cols].to_numpy()
        y = self.encoded_df[y_Col].to_numpy()
        
        return X, y

    ''' REMOVE LATER '''
    def _stratified_split_encoded_df_old(self, test_size=0.3):
        # Splits the data into train and test dataframes (Using Stratified Sampling)
        n_samples = int(test_size * len(self.encoded_df)/len(np.unique(self.encoded_df[self.encoded_df.columns[-1]])))
        test_df = self.encoded_df.groupby(by=self.encoded_df.columns[-1], group_keys=False).apply(lambda x: x.sample(n_samples))

        merged_result = pd.merge(self.encoded_df, test_df, on=list(self.encoded_df.columns), how='left', indicator=True)
        train_df = merged_result[merged_result['_merge'] == 'left_only'].drop(columns=['_merge'])

        return train_df, test_df

    def _stratified_split_encoded_df(self, test_size=0.3):
        # Initialize empty lists to store indices for train and test samples
        test_indices = []
        
        # Iterate over each class in the label column
        for label in np.unique(self.encoded_df[self.encoded_df.columns[-1]]):
            # Getting the classes for each label
            class_subset = self.encoded_df[self.encoded_df[self.encoded_df.columns[-1]] == label]
            # Calculate the number of samples to take from the current class
            n_samples = int(np.ceil(test_size * len(class_subset))) 
            # Get the sample indices
            class_test_indices = class_subset.sample(n=n_samples, replace=False).index
            # Store test indices
            test_indices.extend(class_test_indices.tolist())  
    
        # Create train and test dataframes from the indices
        test_df = self.encoded_df.loc[test_indices]
        train_df = self.encoded_df.drop(test_indices)
    
        return train_df, test_df

    def train_test_split(self, test_size=0.3, shuffle=False):
        # Check if the test_size if valid
        if test_size > 1 or test_size < 0:
            raise Exception("Invalid Test Size Proportion (Must be between 0 - 1)")

        # Shuffles the Data
        if (shuffle):
            self.encoded_df = self.encoded_df.sample(frac = 1)

        # Applying Stratified Sampling
        encoded_train_df, encoded_test_df = self._stratified_split_encoded_df(test_size=test_size)

        # Getting columns
        X_Cols = self.encoded_df.columns[0:-1]
        y_Col = self.encoded_df.columns[-1]

        # Splitting the data into training and testing sets
        X_Train, y_Train = encoded_train_df[X_Cols].to_numpy(), encoded_train_df[y_Col].to_numpy()
        X_Test, y_Test = encoded_test_df[X_Cols].to_numpy(), encoded_test_df[y_Col].to_numpy()
        
        # Returning the sets
        return X_Train, X_Test, y_Train, y_Test

    ''' Shuffle data is not being used '''
    def _shuffle_data(self):
        # Note: The array[rand] actually calls the special method __getitem__

        # Creating a new order
        rand = np.arange(len(self.data))
        np.random.shuffle(rand)
            
        # Rearranges the data / target arrays
        self.data = self.data[rand]
        self.target = self.target[rand]
    
    def K_Fold_CV(self, total_folds=3, model=DecisionTree, *args, **kwargs):
        # Performs a K-Fold Cross Validation

        # Length of the Data
        n = self.target.size

        # Number of folds to perform
        k = total_folds

        # nfold -> size / length of each subset / fold
        nfold = n // k
    
        # List to store all the calculated accuracies
        accuracies = []

        # Getting the indices for the data (will have as many as the length of the dataset)
        indices = np.arange(n)
        np.random.shuffle(indices)

        for i in range(k):
            # Getting the test / train indices of the current fold
            test_indices = indices[i*nfold : (i+1)*nfold]
            train_indices = np.concatenate([indices[: i * nfold], indices[(i + 1) * nfold:]])

            # Splitting the data for each new fold
            X_Train, y_Train = self.data[train_indices], self.target[train_indices]
            X_Test, y_Test = self.data[test_indices], self.target[test_indices]

            # Trainning and Evaluating the Model for each new fold
            new_model = model(*args, **kwargs)
            new_model.fit(X_Train, y_Train)
            predictions = new_model.predict(X_Test)
            accuracies.append(self._Calculate_Accuracy(y_Test, predictions))

        # Returning the average accuracy obtained
        return np.mean(accuracies)

    def _Calculate_Accuracy(self, y_Test, y_Predicted):
        # Calculates the Accuracy given the predictions and their actual values
        return sum(y_Test == y_Predicted) / len(y_Test)

    """ Estimate Holdout """
    def Estimate_Holdout(self, model=DecisionTree, test_size=0.3, *args, **kwargs):
        # Splitting the Data
        X_Train, X_Test, y_Train, y_Test = self.train_test_split(test_size)

        # Creating a new Model
        dt = model(*args, **kwargs)

        # Train the Model
        dt.fit(X_Train, y_Train)

        # Make Predictions
        y_Predicted = dt.predict(X_Test)
        
        # Calculates and Returns the Accuracy of the Model
        return self._Calculate_Accuracy(y_Test, y_Predicted)
