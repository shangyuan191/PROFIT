import pandas as pd 
import polars as pl 
import numpy as np
import os
from glob import glob

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,  roc_curve,auc, roc_auc_score, r2_score

from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
class CNG:
    train_path = "./data/train.parquet"
    test_path = "./data/test.parquet"
    lagged_path = "./data/lags.parquet"
    responders_path = "./data/responders.csv"
    features_path = "./data/features.csv"

def parquet_sorting(path):
    parts = path.split('/')
    partition_id = int(parts[-2].split('=')[1])
    parquet_num = int(parts[-1].split('-')[1].split('.')[0])
    return (partition_id, parquet_num)

class Pre_procesing():
    def __init__(self, df, drop_cols, null_cols):
        self.df = df 
        self.drop_cols = drop_cols
        self.null_cols = null_cols
        print(f'Data: {df.shape}')
        self._main_preprocesing()

    def get_dataframe(self):
        return self.df

    def _main_preprocesing(self):
        self._drop_columns()
        self._drop_null_cols()
        self._drop_null_rows()
        self._drop_dublicates()
    
    def _drop_dublicates(self):
        print(f'Dublicate Values:{self.df.is_duplicated().sum()}')

    def _drop_columns(self):
        # drop repsonders
        self.df = self.df.drop(self.drop_cols)
        print(f'Data Shape(Drop_columns):{self.df.shape}')

    def _drop_null_cols(self):
        # drop null cols 
        self.df = self.df.drop(self.null_cols)
        print(f'Data Shape(Drop_null_cols):{self.df.shape}')

    def _drop_null_rows(self):
        # idenitfy null rows 
        null_counts = self.df.null_count()
        maxrows = max(null_counts.row(0))
        print(f'Drop percentage:{round((maxrows/ self.df.shape[0])*100, 3)}% ')
        self.df = self.df.drop_nulls()
        print(f'Data Shape(Drop_null_rows):{self.df.shape}')

def eval_metrics(y_test, y_pred):
    # Evaluate the model's accuracy
    con_matr = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix\n {con_matr}')
    print(f'{classification_report(y_test, y_pred)}')

import numpy as np
from sklearn.metrics import r2_score

def time_series_error_metrics(y_true, y_pred, weights):
    """
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Mean Absolute Percentage Error (MAPE)
        - Symmetric Mean Absolute Percentage Error (sMAPE)
        - Mean Directional Accuracy (MDA)
        - R-squared (R2)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)    
    weights = np.array(weights) / np.sum(weights)  # Normalize weights to sum to 1
    
    # Compute weighted errors
    mae = np.sum(weights * np.abs(y_true - y_pred))
    mse = np.sum(weights * (y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # Weighted MAPE (avoid division by zero by handling small values of y_true)
    mape = np.sum(weights * np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Weighted sMAPE (avoid division by zero by handling small values in the denominator)
    smape = np.sum(weights * (2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))) * 100

    # Mean Directional Accuracy (MDA)
    directional_accuracy = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    mda = directional_accuracy * 100

    # Weighted R-squared
    r2 = r2_score(y_true, y_pred, sample_weight=weights)
    
    # Return results as a dictionary
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MDA (%)": mda,
        "R2": r2,
    }
    
    return metrics

def ML_models(models, names, X_train, y_train, X_test, y_test, train_weights, test_weights, type='reg'):
    #Variables
    y_predictions =[]

    # Comparison of the ML models
    for model, name in zip(models, names):
        print(f'====== {name} =====')
        # Train the decision tree on the training data
        model.fit(X_train, y_train, sample_weight=train_weights)
        # predictions on the test data
        y_pred = model.predict(X_test)
        #Store the prediction
        y_predictions.append( (str(name), y_pred) )
        #Evaluation Metrics of the model
        if type=='reg':
            error_dict = time_series_error_metrics(y_true=y_test, y_pred=y_pred, weights = test_weights)
            print(error_dict)
        else: 
            eval_lst = eval_metrics(y_true=y_test, y_pred=y_pred, weights = test_weights)
            print(eval_lst)


    return y_predictions, model

features_path = './data/features.csv'
respondesr_path = './data/responders.csv'

target = 'responder_6'

drop_cols =  ['responder_0', 'responder_1', 'responder_2', 
              'responder_3', 'responder_4', 'responder_5', 'responder_7', 
              'responder_8']

null_cols = ['feature_00', 'feature_01', 'feature_02', 'feature_03', 
             'feature_04', 'feature_21', 'feature_26', 'feature_27', 
             'feature_31']

features = pl.read_csv(features_path)
responders = pl.read_csv(respondesr_path)

cng = CNG
# train_paths = sorted(glob(os.path.join(cng.train_path, "*/*")), key=parquet_sorting)
# print(train_paths)
# train_dfs = [pl.read_parquet(path) for path in tqdm(train_paths, desc="Loading parquet files")]
# print(len(train_dfs))
# for train_df in train_dfs:
#     print(train_df)
#     print(train_df.columns)
#     print(train_df.shape)
#     print()
# train_df = pl.concat(train_dfs).to_pandas()
# print(f"all data shape: {train_df.shape}")
# train_df.to_parquet('./data/all_data.parquet',index=False)
# print("all_data saved")
train_df=pd.read_parquet('./data/all_data.parquet')
print(f"all data shape: {train_df.shape}")
print(f"all data columns: {train_df.columns}")
print(f"all data head: {train_df.head()}")
# data_pp = Pre_procesing(df=train_df, drop_cols=drop_cols, null_cols=null_cols)
# train_df_new = data_pp.get_dataframe()

# train_df_new = train_df_new.to_pandas()
# train_df_new['responder_6_new'] = train_df_new['responder_6'].shift(-1)
# train_df_new['feature_new'] = (train_df_new['responder_6_new'] > train_df_new['responder_6']).astype(int)
# train_df_new = train_df_new.drop(['responder_6_new'], axis=1) # drop the column since are the sam as close col

# sample_weights = train_df_new.weight
# print(type(sample_weights))
# train_df_new = train_df_new.drop(['weight', 'date_id', 'time_id'], axis=1)

# # Calculate mean and standard deviation
# means, stds = train_df_new.mean(), train_df_new.std()
# # Standardize
# standardized_df = (train_df_new - means) / stds

# X = standardized_df.drop(columns=['responder_6'])
# y = train_df_new['responder_6']
# X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, sample_weights, 
#                                                                                  test_size=0.2, random_state=42)
# print(type(X_train), type(X_test), type(y_train), type(y_test), type(weights_train), type(weights_test))
# print(f'X_train{X_train.shape}, X_test{X_test.shape},\ny_train{y_train.shape}, y_test{y_test.shape}')


# sample_weights_df=sample_weights.to_frame(name='weight')
# sample_weights_df.to_parquet('./data/sample_weights.parquet',index=False)
# print("Sample Weights saved")

# X_train.to_parquet('./data/X_train.parquet',index=False)
# print("X_train saved")
# X_test.to_parquet('./data/X_test.parquet',index=False)
# print("X_test saved")
# y_train_df = y_train.to_frame(name='y_train')
# y_train_df.to_parquet('./data/y_train.parquet',index=False)
# print("y_train saved")
# y_test_df = y_test.to_frame(name='y_test')
# y_test_df.to_parquet('./data/y_test.parquet',index=False)
# print("y_test saved")
# weights_train_df = weights_train.to_frame(name='weights_train')
# weights_train_df.to_parquet('./data/weights_train.parquet',index=False)
# print("weights_train saved")
# weights_test_df = weights_test.to_frame(name='weights_test')
# weights_test_df.to_parquet('./data/weights_test.parquet',index=False)
# print("weights_test saved")

# sample_weight=pd.read_parquet('./data/sample_weights.parquet').squeeze()
# print(type(sample_weight))
# print(sample_weight.shape)
# print(sample_weight.name)
# print(sample_weight.head())
X_train=pd.read_parquet('./data/X_train.parquet')
X_test=pd.read_parquet('./data/X_test.parquet')
y_train=pd.read_parquet('./data/y_train.parquet').squeeze()
y_test=pd.read_parquet('./data/y_test.parquet').squeeze()
weights_train=pd.read_parquet('./data/weights_train.parquet').squeeze()
weights_test=pd.read_parquet('./data/weights_test.parquet').squeeze()
print(type(X_train), type(X_test), type(y_train), type(y_test), type(weights_train), type(weights_test))
print(f'X_train{X_train.shape}, X_test{X_test.shape},\ny_train{y_train.shape}, y_test{y_test.shape}, \nweights_train{weights_train.shape}, weights_test{weights_test.shape}')
print(f'X_train column:\n {X_train.columns}, X_test column:\n {X_test.columns}, y_train column:\n {y_train.name}, y_test column:\n {y_test.name}, weights_train column:\n {weights_train.name}, weights_test column:\n {weights_test.name}')
print(f'X_train head:\n {X_train.head()}, X_test head:\n {X_test.head()}, y_train head:\n {y_train.head()}, y_test head:\n {y_test.head()}, weights_train head:\n {weights_train.head()}, weights_test head:\n {weights_test.head()}')

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def r_squared(y_true, y_pred):
    """Custom R-squared metric."""
    ss_res = K.sum(K.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # Total sum of squares
    return 1 - ss_res / (ss_tot + K.epsilon())  # Add epsilon to prevent division by zero

model=XGBRegressor()
model.fit(X_train, y_train, sample_weight=weights_train)
# predictions on the test data
y_pred = model.predict(X_test)
metrics=time_series_error_metrics(y_true=y_test, y_pred=y_pred, weights = weights_test)
print(metrics)
# eval_metrics(y_test, y_pred)
# # Define the number of input and output nodes
# input_dim = X.shape[1]  # Number of features (79)
# bottle_neck = 8
# # Set custom learning rate
# optimizer = Adam(learning_rate=0.01)

# # Autoencoder Architecture
# input_layer = Input(shape=(input_dim,))
# encoded = Dense(64, activation='relu')(input_layer)  # Encoder
# encoded = Dense(32, activation='relu')(encoded)
# encoded = Dense(bottle_neck, activation='relu')(encoded)      # Bottleneck layer (feature extraction)
# decoded = Dense(32, activation='relu')(encoded)      # Decoder
# decoded = Dense(64, activation='relu')(decoded)
# decoded = Dense(input_dim, activation='linear')(decoded)  # Output layer

# # Create the model
# autoencoder = Model(inputs=input_layer, outputs=decoded)

# # Compile the model
# autoencoder.compile(optimizer=optimizer, loss='mae',metrics=[r_squared])

# # Define EarlyStopping
# early_stopping = EarlyStopping(
#     monitor='val_r_squared',    # Monitor validation loss
#     mode = 'max',
#     patience=1,            # Number of epochs to wait for improvement
#     min_delta=0.001,       # Minimum change to qualify as an improvement
#     restore_best_weights=True  # Restore weights from the best epoch
# )

# history = autoencoder.fit(
#     X_train.values, X_train.values,
#     epochs=100,
#     batch_size=512,
#     validation_split=0.2,
#     callbacks=[early_stopping]
# )

# encoder_ = Model(inputs=autoencoder.input, outputs=encoded)

# # Generate the reduced features
# X_train_encoded = encoder_.predict(X_train)
# X_test_encoded = encoder_.predict(X_test)

# print("Original shape:", X_train.shape)
# print("Encoded shape:", X_train_encoded.shape)

# ml_models = [LGBMRegressor(), # n_estimators=150, learning_rate=0.05, max_depth=7
#              ElasticNet()  , # alpha=0.1, l1_ratio=0.5
#              XGBRegressor()] # n_estimators=150

# # Evauation Metrics of the models
# y_predictions_MC, xgb_model = ML_models(
#     models=ml_models, names=['LGBMRegressor', 'ElasticNet', 'Xgb'], 
#     X_train=X_train_encoded, y_train=y_train, 
#     X_test=X_test_encoded, y_test=y_test,
#     train_weights=weights_train,
#     test_weights=weights_test, 
#     type='reg' )
