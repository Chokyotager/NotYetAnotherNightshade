import os
import math
import random

from tqdm import tqdm
import csv

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

# Only classification problems
datasets = [x for x in os.listdir("vina_latents/")]

for dataset in tqdm(datasets):

    fig, ax = plt.subplots()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    dataset_name = dataset.replace(".tsv", "")
    data = list(csv.reader(open("vina_latents/" + dataset), delimiter="\t"))

    random.Random(0).shuffle(data)

    split = math.ceil(0.9 * len(data))

    # Define
    training_set = data[:split]
    testing_set = data[split:]

    training_latent_space = np.array([[float(y) for y in x[3:]] for x in training_set])
    training_ground = [float(x[2]) for x in training_set]

    training_latent_space[training_latent_space == -np.inf] = -1e8
    training_latent_space[training_latent_space == np.inf] = 1e8

    #gamma = "auto"

    # RandomForestRegressor(max_depth=None)
    # SVR()

    #clf = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True), SVR(gamma="scale"))
    clf = make_pipeline(StandardScaler(), NuSVR(nu=0.5, C=0.5, cache_size=2048))
    clf.fit(training_latent_space, training_ground)

    testing_latent_space = np.array([[float(y) for y in x[3:]] for x in testing_set])
    testing_ground = [float(x[2]) for x in testing_set]

    testing_latent_space[testing_latent_space == -np.inf] = -1e8
    testing_latent_space[testing_latent_space == np.inf] = 1e8

    model_score = clf.score(testing_latent_space, testing_ground)

    # Predict
    predicted_scores = clf.predict(testing_latent_space)
    df = pd.DataFrame({"Actual Vina score / kcal mol⁻¹": testing_ground, "Predicted Vina score / kcal mol⁻¹": predicted_scores})

    grid = sns.jointplot(data=df, x="Actual Vina score / kcal mol⁻¹", y="Predicted Vina score / kcal mol⁻¹", kind="reg", color="#572a89")
    grid.ax_joint.plot(testing_ground, testing_ground, color="darkred")

    #plt.suptitle(dataset_name + " (n = {})".format(len(testing_set)))
    #ax.tick_params(reset=True, color="black", labelcolor="black", direction="out", which="both", length=5, width=1, top=False, right=False)

    grid.ax_joint.set_xlabel("Actual Vina score / kcal mol⁻¹", fontsize=14)
    grid.ax_joint.set_ylabel("Predicted Vina score / kcal mol⁻¹", fontsize=14)

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_edgecolor("black")

    mae = metrics.mean_absolute_error(testing_ground, predicted_scores)
    rho, p_value = stats.spearmanr(testing_ground, predicted_scores)

    print(dataset_name)
    print("R^2:", model_score)
    print("MAE:", mae)
    print("Spearman's rho:", rho)

    plt.savefig("results/" + dataset_name + ".png", dpi=300)
