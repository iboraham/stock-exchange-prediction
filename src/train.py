from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import numpy as np
import pandas as pd
import datatable as dt
from sklearn.model_selection import GroupKFold

from tqdm import tqdm
from random import choices

import keras_tuner as kt

from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

from utils import *

# Load
TRAINING = True
USE_FINETUNE = True
FOLDS = 5
SEED = 42

train = dt.fread(
    '../db/train_files/stock_prices.csv').to_pandas()
train = prep_prices(train)
train = reduce_mem_usage(train)

features = ["Date", "SecuritiesCode", "Open", "High", "Low", "Close", "Volume"]

X = train[features].values
y = train['Target'].values  # Multitarget


autoencoder, encoder = create_autoencoder(X.shape[-1], y.shape[-1], noise=0.1)
encoder.load_weights('../input/jsmpdata-encoder/encoder.hdf5')
encoder.trainable = False

# Tune


def model_fn(hp): return create_model(hp, X.shape[-1], y.shape[-1], encoder)


tuner = CVTuner(
    hypermodel=model_fn,
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective('val_auc', direction='max'),
        num_initial_points=4,
        max_trials=20))


if TRAINING:
    gkf = PurgedGroupTimeSeriesSplit(n_splits=FOLDS, group_gap=20)
    splits = list(gkf.split(y, groups=train['date'].values))
    tuner.search((X,), (y,), splits=splits, batch_size=4096, epochs=100, callbacks=[
                 EarlyStopping('val_auc', mode='max', patience=3)])
    hp = tuner.get_best_hyperparameters(1)[0]
    pd.to_pickle(hp, f'./models/best_hp_{SEED}.pkl')
    for fold, (train_indices, test_indices) in enumerate(splits):
        model = model_fn(hp)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=4096, callbacks=[
                  EarlyStopping('val_auc', mode='max', patience=10, restore_best_weights=True)])
        model.save_weights(f'./models/model_{SEED}_{fold}.hdf5')
        model.compile(Adam(hp.get('lr')/100), loss='binary_crossentropy')
        model.fit(X_test, y_test, epochs=3, batch_size=4096)
        model.save_weights(f'./models/model_{SEED}_{fold}_finetune.hdf5')
    tuner.results_summary()
