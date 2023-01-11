#
# python3 decode_latent.py <input latent TSV> <output file>
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
    print("Wrong usage, please use:\npython3 decode_latent.py <input latent TSV> <output file>")
    exit()

input_latents = sys.argv[1]
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

print("Decoding latents using the save {}".format(save))

input_latents = open(input_latents).read().split("\n")
latents = [[float(y) for y in x.split("\t")[1:]] for x in input_latents if len(x.split("\t")) > 1]
out = decoder.predict([latents], batch_size=1)[0]

writable = list()

for i in range(len(latents)):

    current_latent = input_latents[i].split("\t")

    appendable = [current_latent[0]] + [str(x) for x in out[i].tolist()]
    writable.append(appendable)

open(output_file, "w+").write("\n".join(["\t".join(x) for x in writable]))
