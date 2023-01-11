import os
import math
import random

from tqdm import tqdm
import csv
import json

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

#sns.set_style("whitegrid", {"axes.grid": False})

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR

import sklearn.metrics as metrics
import xgboost as xgb

cross_validation_runs = 5

all_results = dict()

# Only classification problems
datasets = [x for x in os.listdir("gnina_latents/")]

for dataset in tqdm(datasets):

    dataset_name = dataset.replace(".txt.tsv", "")
    data = list(csv.reader(open("gnina_latents/" + dataset), delimiter="\t"))

    data = [x for x in data if "no result" not in x[2] and x[2] != ""]

    usable_set = data

    random.Random(0).shuffle(usable_set)
    results = {"R2": list(), "MAE": list(), "Rho": list()}

    for i in range(cross_validation_runs):

        print(i)

        # Define
        bottom = math.ceil(i/cross_validation_runs * len(usable_set))
        top = math.ceil((i + 1)/cross_validation_runs * len(usable_set))

        training_set = usable_set[:bottom] + usable_set[top:]
        testing_set = usable_set[bottom:top]

        training_latent_space = np.array([[float(y) for y in x[3:]] for x in training_set])
        training_ground = [float(x[2]) for x in training_set]

        training_latent_space[training_latent_space == -np.inf] = -1e8
        training_latent_space[training_latent_space == np.inf] = 1e8

        clf = make_pipeline(StandardScaler(), NuSVR(nu=0.5, C=0.5, cache_size=2048, max_iter=100000))
        clf.fit(training_latent_space, training_ground)

        testing_latent_space = np.array([[float(y) for y in x[3:]] for x in testing_set])
        testing_ground = [float(x[2]) for x in testing_set]

        testing_latent_space[testing_latent_space == -np.inf] = -1e8
        testing_latent_space[testing_latent_space == np.inf] = 1e8

        model_score = clf.score(testing_latent_space, testing_ground)

        # Predict
        predicted_scores = clf.predict(testing_latent_space)

        mae = metrics.mean_absolute_error(testing_ground, predicted_scores)
        rho, p_value = stats.spearmanr(testing_ground, predicted_scores)

        results["R2"].append(model_score)
        results["MAE"].append(mae)
        results["Rho"].append(rho)

    all_results[dataset_name] = {"R2": results["R2"], "MAE": results["MAE"], "Rho": results["Rho"]}
    print(json.dumps(all_results))
