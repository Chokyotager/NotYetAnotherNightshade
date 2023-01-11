import os
import time
import json
import loading
import gc

import csv

import random
import math

import numpy as np
import scipy as sp
from tqdm import tqdm

import logging

from data import BondType

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

config = json.loads(open("config.json").read())

print("Reading the dataset...")
all_smiles = list(csv.reader(open(config["scores"]), delimiter="\t"))

if config["shuffle-seed"] != None:

    random.Random(config["shuffle-seed"]).shuffle(all_smiles)

print("Initialising the dataset...")

def get_data (smiles):

    # Parse the graphs
    returnable = list()

    for smile in tqdm(smiles):

        if smile[0] == "":
            continue

        molecule_data = smile[1].split(",")

        fingerprint_data = molecule_data[0]
        regression_data = [0 if x == "" else float(x) for x in molecule_data[1:]]

        fingerprint = [int(x) for x in fingerprint_data]

        # CIS TRANS chemistry
        graph = loading.get_data(smile[0], apply_paths=False, parse_cis_trans=False)

        if len(graph[0]) > config["node-cutoff"]:
            continue

        x, a, e = loading.convert(*graph, bonds=[
            BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED])

        fingerprint = list(fingerprint) + regression_data

        returnable.append([(x, a, e), fingerprint])

    return returnable

data = get_data(all_smiles[:config["truncation"]])
gc.collect()

print("Dataset initialised! {} graphs have been loaded.".format(len(data)))

from spektral.data import Dataset, Graph

class SMILESDataset (Dataset):

    def __init__ (self, training=False, all_data=False, **kwargs):

        if all_data:
            assert not training

        self.training = training
        self.all_data = all_data

        super().__init__(**kwargs)

    def download (self):

        # No download function
        pass

    def get_dataset (self):

        cutoff = math.floor((1 - config["validation-ratio"]) * len(data))

        if self.all_data:
            adj_data = data

        else:

            if self.training:
                adj_data = data[:cutoff]

            else:
                adj_data = data[cutoff:]

        return adj_data

    def read (self):

        adj_data = self.get_dataset()
        returnable = list()

        for adj_datapoint in adj_data:

            x, a, e = adj_datapoint[0]
            fingerprint = adj_datapoint[1]

            returnable.append(Graph(x=np.array(x), a=np.array(a), e=np.array(e), y=np.array(fingerprint)))

        return returnable
