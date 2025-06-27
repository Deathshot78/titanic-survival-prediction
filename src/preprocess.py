import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch

def preprocess_data(train_path='data/train.csv', test_path='data/test.csv'):
    """
    Loads, cleans, and feature-engineers the Titanic dataset.

    This function handles:
    - Loading train and test CSVs.
    - Engineering features like Title, FamilySize, IsAlone, and Deck.
    - Imputing missing values for Age, Fare, and Embarked.
    - Returning separate, processed dataframes for training and prediction,
      along with metadata useful for modeling.

    Args:
        train_path (str): The file path for the training data.
        test_path (str): The file path for the test data.

    Returns:
        tuple: A tuple containing:
            - X_full (pd.DataFrame): The full processed training feature set.
            - y_full (pd.Series): The full training target variable.
            - X_predict (pd.DataFrame): The full processed test feature set.
            - test_passenger_ids (pd.Series): The PassengerIds for the test set.
            - numerical_features (list): A list of numerical column names.
            - categorical_features (list): A list of categorical column names.
    """
    print("--- Loading and Preprocessing Data ---")
    try:
        train_df_orig = pd.read_csv(train_path)
        test_df_orig = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find {e.filename}. Please ensure train.csv and test.csv are in the correct directory.")
        raise

    test_passenger_ids = test_df_orig['PassengerId']
    y_full = train_df_orig['Survived'].copy()
    
    combined_df = pd.concat([train_df_orig.drop('Survived', axis=1), test_df_orig], ignore_index=True)

    # Feature Engineering
    def engineer_features(df):
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
        common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
        df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Other')
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'U')
        return df

    combined_df = engineer_features(combined_df)

    # Imputation 
    combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)

    # First, calculate the grouped medians. This might have NaNs for groups where all ages are missing.
    age_median_map = combined_df.groupby(['Pclass', 'Title'])['Age'].median()
    # Then, calculate a global median to use as a fallback.
    global_age_median = combined_df['Age'].median()
    # Fill any NaNs in the median map itself with the global median.
    age_median_map.fillna(global_age_median, inplace=True)
    # Now, we can safely apply this robust map to fill NaNs in the main dataframe.
    combined_df['Age'] = combined_df.apply(
        lambda row: age_median_map.loc[row['Pclass'], row['Title']] if pd.isnull(row['Age']) else row['Age'],
        axis=1
    )

    # Robust Fare Imputation
    combined_df['Fare'] = combined_df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

    # Final Feature Selection 
    numerical_features = ['Age', 'Fare', 'FamilySize', 'IsAlone']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
    
    combined_df_processed = combined_df[numerical_features + categorical_features]

    # Final check for any remaining NaNs
    if combined_df_processed.isnull().sum().sum() > 0:
        print("Error: NaNs still present after preprocessing!")
        print(combined_df_processed.isnull().sum())
        raise ValueError("Preprocessing failed, NaNs remain in the data.")
    
    # Split back into final training and prediction sets 
    X_full = combined_df_processed.iloc[:len(train_df_orig)]
    X_predict = combined_df_processed.iloc[len(train_df_orig):]

    print("Preprocessing complete.")
    
    return X_full, y_full, X_predict, test_passenger_ids, numerical_features, categorical_features

class TitanicDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32),
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1) if y is not None else None,

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

class TitanicDataModule(pl.LightningDataModule):
    def __init__(self, X_full, y_full, X_predict, batch_size=64, num_workers=0):
        super().__init__()
        self.save_hyperparameters(ignore=['X_full', 'y_full', 'X_predict'])
        self.X_full = X_full
        self.y_full = y_full
        self.X_predict = X_predict
        self.scaler = StandardScaler()

    def setup(self, stage=None):
        # Scale numerical features based on the training set
        self.X_full_scaled = self.scaler.fit_transform(self.X_full)
        self.X_predict_scaled = self.scaler.transform(self.X_predict)

        # Split data for training, validation, and testing
        X_train, X_temp, y_train, y_temp = train_test_split(self.X_full_scaled, self.y_full, test_size=0.3, random_state=42, stratify=self.y_full)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        self.train_ds = TitanicDataset(pd.DataFrame(X_train), y_train)
        self.val_ds = TitanicDataset(pd.DataFrame(X_val), y_val)
        self.test_ds = TitanicDataset(pd.DataFrame(X_test), y_test)
        self.predict_ds = TitanicDataset(pd.DataFrame(self.X_predict_scaled))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

if __name__ == '__main__':
    # This block allows you to run this script directly to test the preprocessing
    print("Running preprocessing script as a standalone test...")
    X_train, y_train, X_test, _, _, _ = preprocess_data()
    print("\nShape of processed training features (X_full):", X_train.shape)
    print("Shape of training labels (y_full):", y_train.shape)
    print("Shape of processed test features (X_predict):", X_test.shape)
    print("\nFirst 5 rows of processed training data:")
    print(X_train.head())