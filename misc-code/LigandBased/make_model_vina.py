import os

from tqdm import tqdm
import csv

import numpy as np
from scipy import stats

import joblib
import math
import random

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import NuSVR

from sklearn.neighbors import KNeighborsRegressor

import sklearn.metrics as metrics
import xgboost as xgb

latents = os.listdir("vina_latents")

for latent in latents:

    print("==========================")
    print(latent)

    # Only classification problems
    data = list(csv.reader(open("vina_latents/" + latent), delimiter="\t"))

    transposed_data = [list(x) for x in zip(*data)]

    random.Random(0).shuffle(data)

    split = math.ceil(0.9 * len(data))

    # Define
    training_set = data[:split]
    testing_set = data[split:]

    print("Training", len(training_set))
    print("Testing", len(testing_set))

    training_latent_space = np.array([[float(y) for y in x[3:]] for x in training_set])
    training_ground = [float(x[2]) for x in training_set]

    training_latent_space[training_latent_space == -np.inf] = -1e8
    training_latent_space[training_latent_space == np.inf] = 1e8

    #gamma = "auto"

    # RandomForestRegressor(max_depth=None)
    # SVR()

    #clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True), SVR(gamma="scale"))
    #clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True), ExtraTreesRegressor(n_estimators=512, max_depth=None, max_features="log2", random_state=0, n_jobs=-1))
    #clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True), xgb.XGBRegressor(n_estimators=1024, gpu_id=0))

    #clf = make_pipeline(StandardScaler(), ExtraTreesRegressor(n_estimators=2048, max_depth=None, max_features="log2", random_state=0, n_jobs=-1))
    clf = make_pipeline(StandardScaler(), NuSVR(nu=0.5, C=0.5, cache_size=2048))
    clf.fit(training_latent_space, training_ground)

    testing_latent_space = np.array([[float(y) for y in x[3:]] for x in testing_set])
    testing_ground = [float(x[2]) for x in testing_set]

    testing_latent_space[testing_latent_space == -np.inf] = -1e8
    testing_latent_space[testing_latent_space == np.inf] = 1e8

    model_score = clf.score(testing_latent_space, testing_ground)
    predicted_scores = clf.predict(testing_latent_space)

    mae = metrics.mean_absolute_error(testing_ground, predicted_scores)
    rho, p_value = stats.spearmanr(testing_ground, predicted_scores)

    print("R^2:", model_score)
    print("MAE:", mae)
    print("Spearman's rho:", rho)

    joblib.dump(clf, latent + "_ETC.joblib")
