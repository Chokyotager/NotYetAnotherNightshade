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
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score

import sklearn.metrics as metrics

cross_validation_runs = 5

latents = "5hk"

all_results = dict()

# Perform cross-validation
datasets = [x for x in os.listdir(latents) if x.startswith("classification-")]

print("\t".join(["Dataset", "AUROC x̄", "AUROC s", "AUPRC x̄", "AUPRC s"]))

for dataset in datasets:

    dataset_name = dataset.replace(".tsv", "").replace("classification-", "")
    data = list(csv.reader(open(latents + "/" + dataset), delimiter="\t"))

    transposed_data = [list(x) for x in zip(*data)]

    training_set = [x for x in data if x[0] == "train"]
    validation_set = [x for x in data if x[0] == "valid"]
    test_set = [x for x in data if x[0] == "test"]

    usable_set = training_set + validation_set + test_set

    # Shuffle
    random.Random(3).shuffle(usable_set)
    results = {"AUROC": list(), "AUPRC": list(), "accuracy": list()}

    for i2 in range(cross_validation_runs):

        bottom = math.ceil(i2/cross_validation_runs * len(usable_set))
        top = math.ceil((i2 + 1)/cross_validation_runs * len(usable_set))

        training_set = usable_set[:bottom] + usable_set[top:]
        testing_set = usable_set[bottom:top]

        transposed_data = [list(x) for x in zip(*data)]
        all_labels = sorted(list(set(transposed_data[3])))

        training_latent_space = np.array([[float(y) for y in x[4:]] for x in training_set])
        training_labels = [all_labels.index(x[3]) for x in training_set]

        training_latent_space[training_latent_space == -np.inf] = -1e8
        training_latent_space[training_latent_space == np.inf] = 1e8

        etc = make_pipeline(RobustScaler(quantile_range=(10, 90), unit_variance=True), ExtraTreesClassifier(n_estimators=512, max_features="log2", n_jobs=-1, random_state=0))
        clf = etc

        #clf = VotingClassifier([("trees1", etc), ("gboost", gboost)], voting="soft", n_jobs=-1)
        #clf = VotingClassifier([("trees1", etc), ("svc", svc)], voting="soft", n_jobs=-1)
        clf.fit(training_latent_space, training_labels)

        testing_latent_space = np.array([[float(y) for y in x[4:]] for x in testing_set])
        testing_labels = [all_labels.index(x[3]) for x in testing_set]

        testing_latent_space[testing_latent_space == -np.inf] = -1e8
        testing_latent_space[testing_latent_space == np.inf] = 1e8

        y_score = clf.predict_proba(testing_latent_space)
        y_test = testing_labels

        # Convert y_test to one-hot
        def create_one_hot(size, index):

            vector = [0] * size
            vector[index] = 1

            return vector

        y_test = np.array([create_one_hot(len(all_labels), x) for x in y_test])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(len(all_labels)):

            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        results["AUROC"].append(roc_auc[0])

        # Strictly not equivalent, but more pessimistic
        auprc = metrics.average_precision_score(y_test[:, 0], y_score[:, 0])
        results["AUPRC"].append(auprc)

        accuracy = metrics.accuracy_score(y_test[:, 0], np.round(y_score[:, 0]))
        results["accuracy"].append(accuracy)

    # Calculate 95% CIs
    auroc_mean = str(np.mean(results["AUROC"]))
    auroc_std = str(np.std(results["AUROC"]))

    auprc_mean = str(np.mean(results["AUPRC"]))
    auprc_std = str(np.std(results["AUPRC"]))

    all_results[dataset_name] = {"AUROC": results["AUROC"], "AUPRC": results["AUPRC"], "accuracy": results["accuracy"]}

    print("\t".join([dataset_name, auroc_mean, auroc_std, auprc_mean, auprc_std]))

print(json.dumps(all_results))
