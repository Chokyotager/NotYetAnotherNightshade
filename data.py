import typing
from collections import namedtuple
from enum import Enum
import enum
from typing import Tuple, Any

import numpy as np
from pandas import DataFrame


@enum.unique
class BondType(Enum):
    """
    Enumeration of bond types
    """
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AMIDE = 4
    AROMATIC = 5

    # don't know what these 3 are supposed to be
    DUMMY = 6
    NOT_CONNECTED = 7
    UNKNOWN = 8

    @classmethod
    def from_short(cls, value):
        return {
            '1': cls.SINGLE,
            '2': cls.DOUBLE,
            '3': cls.TRIPLE,
            'am': cls.AMIDE,
            'ar': cls.AROMATIC,
            'du': cls.DUMMY,
            'nc': cls.NOT_CONNECTED,
            'un': cls.UNKNOWN,
        }[value]


@enum.unique
class AtomType(Enum):
    """
    Enumeration of atom types
    """
    Al = enum.auto()  # aluminum
    As = enum.auto()  # arsenic
    B = enum.auto()  # boron
    Be = enum.auto()  # beryllium
    Br = enum.auto()  # bromine
    C_1 = enum.auto()  # sp1 carbon
    C_2 = enum.auto()  # sp2 carbon
    C_3 = enum.auto()  # sp3 carbon
    C_ar = enum.auto()  # aromatic carbon
    C_cat = enum.auto()  # carbon
    Ca = enum.auto()  # calcium
    Cl = enum.auto()  # chlorine
    Co = enum.auto()  # cobalt
    Cu = enum.auto()  # copper
    Du = enum.auto()  # dummy
    F = enum.auto()  # fluorine
    Fe = enum.auto()  # iron
    H = enum.auto()  # hydrogen
    H_spc = enum.auto()  # hydrogen-water spc model
    H_t3p = enum.auto()  # hydrogen-water tip3p model
    I = enum.auto()  # iodine
    Ir = enum.auto()  # iridium
    K = enum.auto()  # potassium
    Li = enum.auto()  # lithium
    LP = enum.auto()  # lone pair electrons
    Mg = enum.auto()  # magnesium
    N_1 = enum.auto()  # sp1 nitrogen
    N_2 = enum.auto()  # sp2 nitrogen
    N_3 = enum.auto()  # sp3 nitrogen
    N_4 = enum.auto()  # quaternary nitrogen
    N_am = enum.auto()  # amide nitrogen
    N_ar = enum.auto()  # aromatic nitrogen
    N_pl3 = enum.auto()  # trigonal nitrogen note: the sybyl guide claims it's N.p13 but all mol files have N.pl3
    Na = enum.auto()  # sodium
    O_2 = enum.auto()  # sp2 oxygen
    O_3 = enum.auto()  # sp3 oxygen
    O_co2 = enum.auto()  # carboxy oxygen
    O_spc = enum.auto()  # oxygen - water spc model
    O_t3p = enum.auto()  # oxygen - water tip3p model
    Os = enum.auto()  # osmium
    P_3 = enum.auto()  # sp3 phosphorous
    Pt = enum.auto()  # platinum
    Re = enum.auto()  # rhenium
    Rh = enum.auto()  # rhodium
    Ru = enum.auto()  # ruthenium
    S_2 = enum.auto()  # sp2 sulfur
    S_3 = enum.auto()  # sp3 sulfur
    S_o = enum.auto()  # sulfoxide sulfur
    S_O = enum.auto()  # sulfoxide sulfur (alternative version)
    S_o2 = enum.auto()  # sulfone sulfur
    S_O2 = enum.auto()  # sulfone sulfur (alternative version)
    Sb = enum.auto()  # antimony
    Se = enum.auto()  # selenium
    Si = enum.auto()  # silicon
    Te = enum.auto()  # tellurium
    V = enum.auto()  # vanadium
    Zn = enum.auto()  # zinc


EdgeInfo = typing.NamedTuple("EdgeInfo", length=float, type=BondType)
NodeInfo = typing.NamedTuple(
    "NodeInfo", x=float, y=float, z=float, type=AtomType, charge=float, is_ligand=bool)


atom_type_map = {i.name: i for i in AtomType}


def lookup_atom_type(name: str, sub=None, is_aromatic=False, unknown_is_dummy=False) -> AtomType:
    if name in atom_type_map:
        return atom_type_map[name]
    if sub is None and is_aromatic:
        if name + "_ar" in atom_type_map:
            return atom_type_map[name + "_ar"]

    if sub is not None:
        sub = sub.lower()
        name += "_" + sub
        if name in atom_type_map:
            return atom_type_map[name]
    else:
        if name + "_1" in atom_type_map:
            return atom_type_map[name + "_1"]
        if name + "_2" in atom_type_map:
            return atom_type_map[name + "_2"]
        if name + "_3" in atom_type_map:
            return atom_type_map[name + "_3"]

    if unknown_is_dummy:
        return AtomType.Du
    raise ValueError(f"Unknown atom type: {name}")


def lookup_bond_type(order: float) -> BondType:

    if order == 1:
        return BondType.SINGLE

    elif order == 2:
        return BondType.DOUBLE

    elif order == 3:
        return BondType.TRIPLE

    elif order == 4:
        return BondType.AMIDE

    elif order == 5:
        return BondType.AROMATIC
    elif order == -1:
        return BondType.NOT_CONNECTED

    else:
        if 1 < order < 2:
            return BondType.AROMATIC
        else:
            raise ValueError(f"Unknown bond type: {order}")
