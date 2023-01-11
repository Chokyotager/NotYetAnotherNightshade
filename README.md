<div align="center">
  <br />
  <p>
    <a href="https://github.com/Chokyotager/NotYetAnotherNightshade"><img src="/art/NYAN.png" alt="banner" /></a>
  </p>
  <br />
  <p>
  </p>
</div>

## About
**NotYetAnotherNightshade** (NYAN) is a graph variational encoder as described in the manuscript "Variational graph encoders: a surprisingly effective generalist algorithm for holistic computer-aided drug design".

It allows for the embedding of molecules into a continuous latent space, and subsequent surrogate model training for molecular property prediction not limited to drug design and other chemistry applications.

The latent space method as described can also be used to perform highly accelerated, very high throughput virtual screening for computer-aided drug discovery of up to a few billion compounds.

This repository contains the code we used in training of new encoders, construction of surrogate models, and latent space potentiation as described in the manuscript. We have also included utility tools for decoding and encoding molecules, so that you can fit and train your own surrogate models.

## Paper

![Figure abstract](https://github.com/Chokyotager/NotYetAnotherNightshade/blob/main/art/abstract.png?raw=true)
Please read the paper for more details

## Installation
```sh
git clone https://github.com/Chokyotager/NotYetAnotherNightshade.git
cd NotYetAnotherNightshade
```

```sh
conda create -y --name nyan python=3.9
conda activate nyan
conda install -y --file requirements.txt
```

## Usage

### Conversion of SMILES to latent space
This tool converts any SMILES molecule into a continuous mathematical space of 64 dimensions.

Please use "encode_smiles.py".

`python3 encode_smiles.py <input SMILES file> <output file>`

The output will be tab-delimited. You can then use this latent space to train surrogate models to your liking. I.e. you can use SKLearn to build ExtraTrees regressors/classifiers to predict ADMET properties.

You can look into datasets/example.smi for an example input file.

### Conversion of latent space into molecular fingerprints
This tool does the opposite and converts a 64-dimensional latent space into molecular fingerprints and Mordred descriptors.

`python3 decode_latent.py <input latent TSV> <output file>`

The first column of the input latent TSV is treated as an ID. Subsequent columns are of the latent space. In total, there should be 65 columns in your input file. All elements should be delimited by tabs. An example of an input file that you can feed in here is the output from `encode_smiles.py`.

The output will be tab-delimited. You can use this to infer molecular properties or match against a known fingerprint database.

You can look into datasets/example_latents.tsv for an example input file.

### Training your own encoder
If you want to train your own model, please edit `config.json`.

You can then train your model using `python3 train.py`

### Other experiments done in the paper
We haven't gotten around to build user-friendly tools for molecular potentiation yet. The code that was used in the paper that was used to do this can be found in misc-code/NYAN-potentiator.

We have also added one in misc-code/referfence-encoding-decoding for latent space and FP searching. To use the stuff in misc-code/, you would probably have to change the code directly.

## Contribution
Users are welcome to contribute to the development of the model through pull-requests.

Our future direction is to make this tool as user-friendly as possible, and to release individual surrogate models that we have trained. Please continue to check back on this repo for the latest updates!

## Maintenance
The current project is maintained by Hilbert Lam and Robbe Pincket. Correspondence can be found in the manuscript. You can also contact Hilbert / Robbe via email there or (informally) on Discord, using the handles ChocoParrot#8925, Kroppeb#2845 respectively (we're chill people).

## License
License details can be found in the LICENSE file.
