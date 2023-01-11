import os
import warnings
import json
import csv
import math
import random
import loading
import numpy as np
import datetime

from data import BondType

import multiprocessing as mp

from tqdm import tqdm
import scipy as sp

processes = 32
return_amount = 32

config = json.loads(open("config.json").read())

trajectory = json.loads(open("NYAN-potentiator/carcinogen_trajectory.json").read())
reference_latents = open("/data/a/ZINC-clustered-latents.tsv").read().split("\n")

for trajectory_iteration in trajectory:

    latents = trajectory_iteration["latent"]

    def screen_latents (reference_latents, output):

        best_score = 167 + 512
        best_matches = [[None, best_score]]

        for reference_latent in reference_latents:

            latent_entry = reference_latent.split("\t")

            smiles = latent_entry[0]
            latent_variables = [float(x) for x in latent_entry[1:]]

            if len(latent_variables) == 0:
                continue

            # Calculate Euclidean
            distance = math.sqrt(sum([math.pow(latents[i] - latent_variables[i], 2) for i in range(len(latents))]))

            if distance >= best_matches[0][1]:
                continue

            best_matches = ([[smiles, distance]] + best_matches)[:return_amount]

            output.append(best_matches)

    manager = mp.Manager()
    output = manager.list()

    all_processes = list()

    for i in range(processes):

        bottom = math.floor(i / processes * len(reference_latents))
        top = math.floor((i + 1) / processes * len(reference_latents))

        process = mp.Process(target=screen_latents, args=[reference_latents[bottom : top], output])

        all_processes.append(process)

    for process in all_processes:
        process.start()

    for process in tqdm(all_processes):
        process.join()
        process.close()

    best_matches = list()

    for best_match in output:
        best_matches += best_match

    best_matches = sorted(best_matches, key=lambda x: x[1])[:return_amount]

    print("\nRESULTS\n===========")

    print(trajectory_iteration["iteration"])
    print(trajectory_iteration["model_scores"])
    print(best_matches[0])
