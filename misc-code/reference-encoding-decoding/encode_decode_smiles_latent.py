import os
import warnings
import json
import csv
import random
import loading
import numpy as np
import datetime
import sys

import math
import multiprocessing as mp
from data import BondType
import logging
from tqdm import tqdm

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from spektral.data import BatchLoader, Dataset, Graph
from spektral import transforms

from extmodel5hk import VAE, NyanEncoder, NyanDecoder, EpochCounter

import scipy as sp
import tensorflow as tf

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)
all_smiles = [sys.argv[1]]

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

save = "saves/ZINC-extmodel5hk-3M"
batch_size = 4
eval_all = True

processes = 32

with strategy.scope():

    encoder = NyanEncoder(latent_dim=64, batched=True)
    decoder = NyanDecoder(fingerprint_bits=679, regression=1613)

    model = VAE(encoder, decoder)
    model.load_weights(save).expect_partial()

print("Evaluating save {}".format(save))

# Initialise dataset
graph_data = list()

for smile in tqdm(all_smiles):

    if smile[0] == "":
        continue

    graph = loading.get_data(smile, only_biggest=True, unknown_atom_is_dummy=True)

    x, a, e = loading.convert(*graph, bonds=[BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED])

    graph = Graph(x=np.array(x), a=np.array(a), e=np.array(e), y=np.array(0))

    graph_data.append([graph, None])

class EvalDataset (Dataset):

    def __init__ (self, **kwargs):
        super().__init__(**kwargs)

    def read (self):
        return [x[0] for x in graph_data]

dataset = EvalDataset()
loader = BatchLoader(dataset, batch_size=batch_size, epochs=1, mask=True, shuffle=False, node_level=False)

predictions = encoder.predict(loader.load())
latents = predictions[0].tolist()

reference_latents = open("/data/a/ZINC-clustered-latents.tsv").read().split("\n")

processes = 32
return_amount = 32

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

for i in range(processes - 1):

    bottom = math.floor(i / 64 * len(reference_latents))
    top = math.floor((i + 1) / 64 * len(reference_latents))

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

for i in range(len(best_matches)):

    best_match = best_matches[i]
    print(i + 1, best_match[0], best_match[1])
