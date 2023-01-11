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

import logging

from tqdm import tqdm

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from spektral.data import BatchLoader, Dataset, Graph
from spektral import transforms

from extmodel5hk import VAE, NyanEncoder, NyanDecoder, EpochCounter

import scipy as sp
import tensorflow as tf

processes = 32

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)
config = json.loads(open("config.json").read())

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

save = "saves/ZINC-extmodel5hk-3M"
batch_size = 32
eval_all = True

with strategy.scope():

    encoder = NyanEncoder(latent_dim=64, batched=True)
    decoder = NyanDecoder(fingerprint_bits=679, regression=1613)

    model = VAE(encoder, decoder)
    model.load_weights(save).expect_partial()

print("Evaluating save {}".format(save))

trajectory = json.loads(open("NYAN-potentiator/trajectory.json").read())

for trajectory_iteration in trajectory:

    latents = trajectory_iteration["latent"]

    out = decoder.predict([latents])[0]

    decoded_fingerprint = out.tolist()[:167 + 512]
    reference_fingerprints = open("NYAN-potentiator/morgan512_maccs.tsv").read().split("\n")[:-1]

    def screen_fingerprint (reference_fingerprints, output):

        best_score = 167 + 512
        best_matches = [[None, best_score]]

        for reference_fingerprint in reference_fingerprints:

            fingerprint_entry = reference_fingerprint.split("\t")

            smiles = fingerprint_entry[0]
            fingerprint = [float(x) for x in list(fingerprint_entry[1])]

            maccs_difference = sum([abs(fingerprint[i] - decoded_fingerprint[i]) for i in range(167)])
            morgan_difference = sum([abs(fingerprint[i] - decoded_fingerprint[i]) for i in range(167, 512)])

            difference = maccs_difference + morgan_difference

            if difference >= best_matches[0][1]:
                continue

            best_matches = ([[smiles, difference]] + best_matches)[:10]

            output.append(best_matches)

    manager = mp.Manager()
    output = manager.list()

    all_processes = list()

    for i in range(processes):

        bottom = math.floor(i / processes * len(reference_fingerprints))
        top = math.floor((i + 1) / processes * len(reference_fingerprints))

        process = mp.Process(target=screen_fingerprint, args=[reference_fingerprints[bottom : top], output])

        all_processes.append(process)

    for process in all_processes:
        process.start()

    for process in tqdm(all_processes):
        process.join()
        process.close()

    best_matches = list()

    for best_match in output:
        best_matches += best_match

    best_matches = sorted(best_matches, key=lambda x: x[1])[:10]

    print(best_matches[0])
