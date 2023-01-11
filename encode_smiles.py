#
# python3 encode_smiles.py <SMILES file> <Output TSV>
#

import os
import sys
import warnings
import json
import csv
import random
import loading
import numpy as np

from data import BondType

import logging

from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

if len(sys.argv) != 3:
    print("Wrong usage, please use:\npython3 encode_smiles.py <SMILES file> <Output TSV>")
    exit()

input_smiles = sys.argv[1]
output_file = sys.argv[2]

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from spektral.data import BatchLoader, Dataset, Graph
from spektral import transforms

from model import VAE, NyanEncoder, NyanDecoder, EpochCounter

import scipy as sp
import tensorflow as tf

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

strategy = tf.distribute.MirroredStrategy()

save = dir_path + "saves/ZINC-extmodel5hk-3M"
batch_size = 1

with strategy.scope():

    encoder = NyanEncoder(latent_dim=64, batched=True)
    decoder = NyanDecoder(fingerprint_bits=679, regression=1613)

    model = VAE(encoder, decoder)
    model.load_weights(save).expect_partial()

print("Generating latents using the save {}".format(save))

all_smiles = [x.split()[0] for x in open(input_smiles).read().split("\n") if len(x.split()) > 1]

# Initialise dataset
graph_data = list()
passed = list()

print("Loading {} molecules".format(len(all_smiles)))

for smile in tqdm(all_smiles):

    if smile[0] == "":
        continue

    try:

        graph = loading.get_data(smile, only_biggest=True, unknown_atom_is_dummy=True)

        x, a, e = loading.convert(*graph, bonds=[BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED])

        graph = Graph(x=np.array(x), a=np.array(a), e=np.array(e), y=np.array(0))

        graph_data.append([graph, None])

        passed.append(smile)

    except Exception as error:

        print("Errored loading SMILES", smile)

class EvalDataset (Dataset):

    def __init__ (self, **kwargs):
        super().__init__(**kwargs)

    def read (self):
        return [x[0] for x in graph_data]

dataset = EvalDataset()
loader = BatchLoader(dataset, batch_size=batch_size, epochs=1, mask=True, shuffle=False, node_level=False)

predictions = encoder.predict(loader.load())
predictions = [[float(y) for y in x] for x in predictions]

writable = list()

for i in range(len(passed)):

    current_smiles = passed[i]

    appendable = [current_smiles] + [str(x) for x in predictions[i]]
    writable.append(appendable)

open(output_file, "w+").write("\n".join(["\t".join(x) for x in writable]))
