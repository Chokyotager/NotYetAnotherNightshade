import os
import math
import random
import json

from tqdm import tqdm
import csv

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import ExtraTreesRegressor

import sklearn.metrics as metrics

cross_validation_runs = 5

latents = "5hk"

all_results = dict()

# Perform cross-validation
datasets = [x for x in os.listdir(latents) if x.startswith("regression-")]

print("\t".join(["Dataset", "R^2 x̄", "R^2 s", "MAE x̄", "MAE s", "Rho x̄", "Rho s"]))

for dataset in datasets:

    dataset_name = dataset.replace(".tsv", "").replace("regression-", "")
    data = list(csv.reader(open(latents + "/" + dataset), delimiter="\t"))

    transposed_data = [list(x) for x in zip(*data)]

    training_set = [x for x in data if x[0] == "train"]
    validation_set = [x for x in data if x[0] == "valid"]
    test_set = [x for x in data if x[0] == "test"]

    usable_set = training_set + validation_set + test_set

    # Shuffle
    random.Random(0).shuffle(usable_set)
    results = {"R2": list(), "MAE": list(), "Rho": list()}

    for i in range(cross_validation_runs):

        bottom = math.ceil(i/cross_validation_runs * len(usable_set))
        top = math.ceil((i + 1)/cross_validation_runs * len(usable_set))

        training_set = usable_set[:bottom] + usable_set[top:]
        testing_set = usable_set[bottom:top]

        training_latent_space = np.array([[float(y) for y in x[4:]] for x in training_set])
        training_ground = [float(x[3]) for x in training_set]

        training_latent_space[training_latent_space == -np.inf] = -1e8
        training_latent_space[training_latent_space == np.inf] = 1e8

        clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True), ExtraTreesRegressor(n_estimators=2048, max_depth=None, max_features="log2", random_state=0, n_jobs=-1))
        clf.fit(training_latent_space, training_ground)

        testing_latent_space = np.array([[float(y) for y in x[4:]] for x in testing_set])
        testing_ground = [float(x[3]) for x in testing_set]

        testing_latent_space[testing_latent_space == -np.inf] = -1e8
        testing_latent_space[testing_latent_space == np.inf] = 1e8

        model_score = clf.score(testing_latent_space, testing_ground)
        predicted_scores = clf.predict(testing_latent_space)

        mae = metrics.mean_absolute_error(testing_ground, predicted_scores)
        rho, p_value = stats.spearmanr(testing_ground, predicted_scores)

        results["R2"].append(model_score)
        results["MAE"].append(mae)
        results["Rho"].append(rho)

    # Calculate 95% CIs
    r2_mean = str(np.mean(results["R2"]))
    r2_std = str(np.std(results["R2"]))

    mae_mean = str(np.mean(results["MAE"]))
    mae_std = str(np.std(results["MAE"]))

    rho_mean = str(np.mean(results["Rho"]))
    rho_std = str(np.std(results["Rho"]))

    all_results[dataset_name] = {"R2": results["R2"], "MAE": results["MAE"], "Rho": results["Rho"]}

    print("\t".join([dataset_name, r2_mean, r2_std, mae_mean, mae_std, rho_mean, rho_std]))

print(json.dumps(all_results))
