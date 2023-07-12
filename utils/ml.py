import os
import sklearn
import seaborn as sns
import numpy as np
import pandas as pd
import re
import pickle
import json
import math
import random
from tqdm import tqdm
import datetime
import time

from PyAstronomy import pyasl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, MissingIndicator

from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    MaxAbsScaler, 
    # RobustScalar,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
    OneHotEncoder, 
    OrdinalEncoder,
    LabelEncoder
)

from sklearn.utils import all_estimators

from sklearn.base import (
    RegressorMixin, 
    ClassifierMixin,
    TransformerMixin
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    auc,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
    classification_report
)

import warnings
import xgboost
import catboost
import lightgbm

import tensorflow as tf

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

removed_classifiers = [
    "ClassifierChain",
    "ComplementNB",
    # "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    # "MLPClassifier",
    "LogisticRegressionCV", 
    "MultiOutputClassifier", 
    "MultinomialNB", 
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
    "CategoricalNB",
    "StackingClassifier",
    "NuSVC",
]

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression", 
    "CCA", 
    "IsotonicRegression", 
    "StackingRegressor",
    "MultiOutputRegressor", 
    "MultiTaskElasticNet", 
    "MultiTaskElasticNetCV", 
    "MultiTaskLasso", 
    "MultiTaskLassoCV", 
    "PLSCanonical", 
    "PLSRegression", 
    "RadiusNeighborsRegressor", 
    "RegressorChain", 
    "VotingRegressor", 
    "QuantileRegressor"
]

CLASSIFIERS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]


REGRESSORS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]

REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
REGRESSORS.append(('CatBoostRegressor', catboost.CatBoostRegressor))

CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
CLASSIFIERS.append(('CatBoostClassifier', catboost.CatBoostClassifier))

TRANSFOMER_METHODS = [
    ("StandardScaler", StandardScaler), 
    ("MinMaxScaler", MinMaxScaler), 
    ("MaxAbsScaler", MaxAbsScaler), 
    # ("RobustScalar", RobustScalar),
    ("Normalizer", Normalizer),
    ("QuantileTransformer", QuantileTransformer),
    ("PowerTransformer", PowerTransformer),
]

def adjusted_rsquared(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))