import json
import os

import smiles_parser

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

properties = json.loads(open(dir_path + "atomic-properties.json").read())

reduction_enum = properties["Reduction enumeration"]
unique_reductions = sorted(list(set(reduction_enum.values())))

def get_data(
        smiles,
        apply_paths=False,
        parse_cis_trans=False,
        only_biggest=False,
        split_ions=False,
        unknown_atom_is_dummy=False
):
    return smiles_parser.parse_smiles(
        smiles,
        apply_paths=apply_paths,
        parse_cis_trans=parse_cis_trans,
        only_biggest=only_biggest,
        split_ions=split_ions,
        unknown_atom_is_dummy=unknown_atom_is_dummy
    )

def convert(x, a, e, path=None, bonds=None):
    def create_one_hot(size, index):

        vector = [0] * size
        vector[index] = 1

        return vector

    converted_x = list()

    # One-hot atom type
    for i in range(len(x)):
        atom_simplified = unique_reductions.index(reduction_enum[x[i]["type"].name])
        atom_one_hot = create_one_hot(len(unique_reductions), atom_simplified)

        converted_x.append(atom_one_hot)

    converted_e = list()

    bond_enum = list(data.BondType) if bonds is None else bonds
    bond_types = {bond: i for i, bond in enumerate(bond_enum)}

    for i in range(len(e)):
        bond = bond_types[e[i]["type"]]
        bond_one_hot = create_one_hot(len(bond_enum), bond)

        converted_e.append(bond_one_hot)

    return converted_x, a, converted_e
