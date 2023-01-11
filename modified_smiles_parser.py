# -*- coding: utf-8 -*-
# Copyright 2018 Peter C Kroon
# modified by Kroppeb (Robbe Pincket) 2022
# more metallole additions added by ChocoParrot (Hilbert Lam) 2022

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Exposes functionality needed for parsing SMILES strings.
"""

import enum
import logging
import re

import networkx as nx

from pysmiles import (add_explicit_hydrogens, remove_explicit_hydrogens, fill_valence)

LOGGER = logging.getLogger("pysmiles")


AROMATIC_ATOMS = "B C N O P S Se As *".split()

ISOTOPE_PATTERN = r'(?P<isotope>[\d]+)?'
ELEMENT_PATTERN = r'(?P<element>b|c|n|o|s|p|bi|ga|ge|sn|sb|pb|ti|zr|se|as|te|\*|[A-Z][a-z]{0,2})'
STEREO_PATTERN = r'(?P<stereo>@|@@|@TH[1-2]|@AL[1-2]|@SP[1-3]|@OH[\d]{1,2}|'\
                  r'@TB[\d]{1,2})?'
HCOUNT_PATTERN = r'(?P<hcount>H[\d]?)?'
CHARGE_PATTERN = r'(?P<charge>(-|\+)(\++|-+|[\d]{1,2})?)?'
CLASS_PATTERN = r'(?::(?P<class>[\d]+))?'
ATOM_PATTERN = re.compile(r'^\[' + ISOTOPE_PATTERN + ELEMENT_PATTERN +
                          STEREO_PATTERN + HCOUNT_PATTERN + CHARGE_PATTERN +
                          CLASS_PATTERN + r'\]$')

@enum.unique
class TokenType(enum.Enum):
    """Possible SMILES token types"""
    ATOM = 1
    BOND_TYPE = 2
    BRANCH_START = 3
    BRANCH_END = 4
    RING_NUM = 5
    EZSTEREO = 6


def _tokenize(smiles):
    """
    Iterates over a SMILES string, yielding tokens.
    Parameters
    ----------
    smiles : iterable
        The SMILES string to iterate over
    Yields
    ------
    tuple(TokenType, str)
        A tuple describing the type of token and the associated data
    """
    organic_subset = 'B C N O P S F Cl Br I * b c n o s p'.split()
    smiles = iter(smiles)
    token = ''
    peek = None
    while True:
        char = peek if peek else next(smiles, '')
        peek = None
        if not char:
            break
        if char == '[':
            token = char
            for char in smiles:
                token += char
                if char == ']':
                    break
            yield TokenType.ATOM, token
        elif char in organic_subset:
            peek = next(smiles, '')
            if char + peek in organic_subset:
                yield TokenType.ATOM, char + peek
                peek = None
            else:
                yield TokenType.ATOM, char
        elif char in '-=#$:.':
            yield TokenType.BOND_TYPE, char
        elif char == '(':
            yield TokenType.BRANCH_START, '('
        elif char == ')':
            yield TokenType.BRANCH_END, ')'
        elif char == '%':
            # If smiles is too short this will raise a ValueError, which is
            # (slightly) prettier than a StopIteration.
            yield TokenType.RING_NUM, int(next(smiles, '') + next(smiles, ''))
        elif char in '/\\':
            yield TokenType.EZSTEREO, char
        elif char.isdigit():
            yield TokenType.RING_NUM, int(char)


def read_smiles(smiles, explicit_hydrogen=False, zero_order_bonds=True,
                reinterpret_aromatic=True):
    """
    Parses a SMILES string.
    Parameters
    ----------
    smiles : iterable
        The SMILES string to parse. Should conform to the OpenSMILES
        specification.
    explicit_hydrogen : bool
        Whether hydrogens should be explicit nodes in the outout graph, or be
        implicit in 'hcount' attributes.
    reinterprit_aromatic : bool
        Whether aromaticity should be determined from the created molecule,
        instead of taken from the SMILES string.
    Returns
    -------
    nx.Graph
        A graph describing a molecule. Nodes will have an 'element', 'aromatic'
        and a 'charge', and if `explicit_hydrogen` is False a 'hcount'.
        Depending on the input, they will also have 'isotope' and 'class'
        information.
        Edges will have an 'order'.
    """
    bond_to_order = {'-': 1, '=': 2, '#': 3, '$': 4, ':': 1.5, '.': 0}
    mol = nx.Graph()
    anchor = None
    idx = 0
    default_bond = 1
    next_bond = None
    ez_stereo_dir = None
    branches = []
    ring_nums = {}
    for tokentype, token in _tokenize(smiles):
        if tokentype == TokenType.ATOM:
            mol.add_node(idx, **parse_atom(token))
            if anchor is not None:
                if next_bond is None:
                    next_bond = default_bond
                if next_bond or zero_order_bonds:
                    ez_stereo = None
                    if ez_stereo_dir is not None:
                        if ez_stereo_dir == "/":
                            ez_stereo = (anchor, idx)
                        else:
                            ez_stereo = (idx, anchor)
                        ez_stereo_dir = None
                    mol.add_edge(anchor, idx, order=next_bond, ez_stereo=ez_stereo)
                next_bond = None
            anchor = idx
            idx += 1
        elif tokentype == TokenType.BRANCH_START:
            branches.append(anchor)
        elif tokentype == TokenType.BRANCH_END:
            anchor = branches.pop()
        elif tokentype == TokenType.BOND_TYPE:
            if next_bond is not None:
                raise ValueError('Previous bond (order {}) not used. '
                                 'Overwritten by "{}"'.format(next_bond, token))
            next_bond = bond_to_order[token]
        elif tokentype == TokenType.RING_NUM:
            if token in ring_nums:
                jdx, order = ring_nums[token]
                if next_bond is None and order is None:
                    next_bond = default_bond
                elif order is None:  # Note that the check is needed,
                    next_bond = next_bond  # But this could be pass.
                elif next_bond is None:
                    next_bond = order
                elif next_bond != order:  # Both are not None
                    raise ValueError('Conflicting bond orders for ring '
                                     'between indices {}'.format(token))
                # idx is the index of the *next* atom we're adding. So: -1.
                if mol.has_edge(idx - 1, jdx):
                    raise ValueError('Edge specified by marker {} already '
                                     'exists'.format(token))
                if idx - 1 == jdx:
                    raise ValueError('Marker {} specifies a bond between an '
                                     'atom and itself'.format(token))
                if next_bond or zero_order_bonds:
                    mol.add_edge(idx - 1, jdx, order=next_bond)
                next_bond = None
                del ring_nums[token]
            else:
                if idx == 0:
                    raise ValueError("Can't have a marker ({}) before an atom"
                                     "".format(token))
                # idx is the index of the *next* atom we're adding. So: -1.
                ring_nums[token] = (idx - 1, next_bond)
                next_bond = None
        elif tokentype == TokenType.EZSTEREO:
            ez_stereo_dir = token
    if ring_nums:
        raise KeyError('Unmatched ring indices {}'.format(list(ring_nums.keys())))

    # Time to deal with aromaticity. This is a mess, because it's not super
    # clear what aromaticity information has been provided, and what should be
    # inferred. In addition, to what extend do we want to provide a "sane"
    # molecule, even if this overrides what the SMILES string specifies?
    cycles = nx.cycle_basis(mol)
    ring_idxs = set()
    for cycle in cycles:
        ring_idxs.update(cycle)
    non_ring_idxs = set(mol.nodes) - ring_idxs
    for n_idx in non_ring_idxs:
        if mol.nodes[n_idx].get('aromatic', False):
            raise ValueError("You specified an aromatic atom outside of a"
                             " ring. This is impossible")

    mark_aromatic_edges(mol)
    fill_valence(mol)
    if reinterpret_aromatic:
        mark_aromatic_atoms(mol)
        mark_aromatic_edges(mol)
        for idx, jdx in mol.edges:
            if ((not mol.nodes[idx].get('aromatic', False) or
                 not mol.nodes[jdx].get('aromatic', False))
                    and mol.edges[idx, jdx].get('order', 1) == 1.5):
                mol.edges[idx, jdx]['order'] = 1

    if explicit_hydrogen:
        add_explicit_hydrogens(mol)
    else:
        remove_explicit_hydrogens(mol)
    return mol


def parse_atom(atom):
    """
    Parses a SMILES atom token, and returns a dict with the information.
    Note
    ----
    Can not deal with stereochemical information yet. This gets discarded.
    Parameters
    ----------
    atom : str
        The atom string to interpret. Looks something like one of the
        following: "C", "c", "[13CH3-1:2]"
    Returns
    -------
    dict
        A dictionary containing at least 'element', 'aromatic', and 'charge'. If
        present, will also contain 'hcount', 'isotope', and 'class'.
    """
    defaults = {'charge': 0, 'hcount': 0, 'aromatic': False}
    if not atom.startswith('[') and not atom.endswith(']'):
        if atom != '*':
            # Don't specify hcount to signal we don't actually know anything
            # about it
            return {'element': atom.capitalize(), 'charge': 0,
                    'aromatic': atom.islower()}
        else:
            return defaults.copy()
    match = ATOM_PATTERN.match(atom)
    if match is None:
        raise ValueError('The atom {} is malformatted'.format(atom))
    out = defaults.copy()
    out.update({k: v for k, v in match.groupdict().items() if v is not None})

    if out.get('element', 'X').islower():
        out['aromatic'] = True

    parse_helpers = {
        'isotope': int,
        'element': str.capitalize,
        'stereo': lambda x: x,
        'hcount': parse_hcount,
        'charge': parse_charge,
        'class': int,
        'aromatic': lambda x: x,
    }

    for attr, val_str in out.items():
        out[attr] = parse_helpers[attr](val_str)

    if out['element'] == '*':
        del out['element']

    if out.get('element') == 'H' and out.get('hcount', 0):
        raise ValueError("A hydrogen atom can't have hydrogens")

    if 'stereo' in out:
        LOGGER.warning('Atom "%s" contains stereochemical information that will be discarded.', atom)

    return out


def parse_hcount(hcount_str):
    """
    Parses a SMILES hydrogen count specifications.
    Parameters
    ----------
    hcount_str : str
        The hydrogen count specification to parse.
    Returns
    -------
    int
        The number of hydrogens specified.
    """
    if not hcount_str:
        return 0
    if hcount_str == 'H':
        return 1
    return int(hcount_str[1:])


def parse_charge(charge_str):
    """
    Parses a SMILES charge specification.
    Parameters
    ----------
    charge_str : str
        The charge specification to parse.
    Returns
    -------
    int
        The charge.
    """
    if not charge_str:
        return 0
    signs = {'-': -1, '+': 1}
    sign = signs[charge_str[0]]
    if len(charge_str) > 1 and charge_str[1].isdigit():
        charge = sign * int(charge_str[1:])
    else:
        charge = sign * charge_str.count(charge_str[0])
    return charge


def mark_aromatic_atoms(mol, atoms=None):
    """
    Sets the 'aromatic' attribute for all nodes in `mol`. Requires that
    the 'hcount' on atoms is correct.
    Parameters
    ----------
    mol : nx.Graph
        The molecule.
    atoms: collections.abc.Iterable
        The atoms to act on. Will still analyse the full molecule.
    Returns
    -------
    None
        `mol` is modified in-place.
    """
    if atoms is None:
        atoms = set(mol.nodes)
    aromatic = set()
    # Only cycles can be aromatic
    for cycle in nx.cycle_basis(mol):
        # All atoms should be sp2, so each contributes an electron. We make
        # sure they are later.
        electrons = len(cycle)
        maybe_aromatic = True

        for node_idx in cycle:
            node = mol.nodes[node_idx]
            element = node.get('element', '*').capitalize()
            hcount = node.get('hcount', 0)
            degree = mol.degree(node_idx) + hcount
            hcount += _hydrogen_neighbours(mol, node_idx)
            # Make sure they are possibly aromatic, and are sp2 hybridized
            if element not in AROMATIC_ATOMS or degree not in (2, 3):
                maybe_aromatic = False
                break
            # Some of the special cases per group. N and O type atoms can
            # donate an additional electron from a lone pair.
            # missing cases:
            #   extracyclic sp2 heteroatom (e.g. =O)
            #   some charged cases
            if element in 'N P As'.split() and hcount == 1:
                electrons += 1
            elif element in 'O S Se'.split():
                electrons += 1
            if node.get('charge', 0) == +1 and not (element == 'C' and hcount == 0):
                electrons -= 1
        if maybe_aromatic and int(electrons) % 2 == 0:
            # definitely (anti) aromatic
            aromatic.update(cycle)
    for node_idx in atoms:
        node = mol.nodes[node_idx]
        if node_idx not in aromatic:
            node['aromatic'] = False
        else:
            node['aromatic'] = True


def mark_aromatic_edges(mol):
    """
    Set all bonds between aromatic atoms (attribute 'aromatic' is `True`) to
    1.5. Gives all other bonds that don't have an order yet an order of 1.
    Parameters
    ----------
    mol : nx.Graph
        The molecule.
    Returns
    -------
    None
        `mol` is modified in-place.
    """
    for cycle in nx.cycle_basis(mol):
        for idx, jdx in mol.edges(nbunch=cycle):
            if idx not in cycle or jdx not in cycle:
                continue
            if (mol.nodes[idx].get('aromatic', False)
                    and mol.nodes[jdx].get('aromatic', False)):
                mol.edges[idx, jdx]['order'] = 1.5
    for idx, jdx in mol.edges:
        if 'order' not in mol.edges[idx, jdx]:
            mol.edges[idx, jdx]['order'] = 1


def _hydrogen_neighbours(mol, n_idx):
    neighbours = mol[n_idx]
    h_neighbours = 0
    for n_jdx in neighbours:
        if (mol.nodes[n_jdx].get('element', '*') == 'H' and
                mol.edges[n_idx, n_jdx].get('order', 1) == 1):
            h_neighbours += 1
    return h_neighbours
