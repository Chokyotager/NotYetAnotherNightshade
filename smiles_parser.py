import modified_smiles_parser as pysmiles
import networkx

from data import NodeInfo, EdgeInfo, lookup_atom_type, lookup_bond_type

def parse_smiles(
        input: str,
        apply_paths=False,
        parse_cis_trans=False,
        only_biggest=False,
        split_ions=False,
        unknown_atom_is_dummy=False
) -> (
        list[map], list[list[int]], list[map], list[tuple]):  # x, a, e, paths

    # Substitutions
    #for replace_from, replace_to in metallole_substitutions.items():
    #    input = input.replace(replace_from, replace_to)

    # networkx graph
    network: networkx.Graph = pysmiles.read_smiles(
        input,
        explicit_hydrogen=True,
        zero_order_bonds=False,
        reinterpret_aromatic=True,
    )

    assert not(only_biggest and split_ions)

    if only_biggest:
        network = network.subgraph(max(networkx.connected_components(network), key=len)).copy()

    if split_ions:
        return [parse_smiles_internal(
            network.subgraph(sub_network).copy(),
            apply_paths=apply_paths,
            parse_cis_trans=parse_cis_trans,
            unknown_atom_is_dummy=unknown_atom_is_dummy
        ) for sub_network in networkx.connected_components(network)]
    else:
        return parse_smiles_internal(
            network,
            apply_paths=apply_paths,
            parse_cis_trans=parse_cis_trans,
            unknown_atom_is_dummy=unknown_atom_is_dummy
        )


def parse_smiles_internal(
        network: networkx.Graph,
        apply_paths,
        parse_cis_trans,
        unknown_atom_is_dummy
):
    nodes = []
    neighbour_lists = {}
    node_map = {}

    for x, (i, node) in enumerate(network.nodes(data=True)):
        nodes.append(node)
        neighbour_lists[i] = []
        node_map[i] = x
        # print(node)

    for x, y, data in network.edges(data=True):
        stereo = data["ez_stereo"] if "ez_stereo" in data else None
        if stereo is None:
            neighbour_lists[x].append((y, data["order"], 0))
            neighbour_lists[y].append((x, data["order"], 0))
        elif stereo[0] == x:
            neighbour_lists[x].append((y, data["order"], 1))
            neighbour_lists[y].append((x, data["order"], -1))
        else:
            neighbour_lists[x].append((y, data["order"], -1))
            neighbour_lists[y].append((x, data["order"], 1))

    # find stereo paths
    stereo_paths = []
    if parse_cis_trans:
        for i, neighbours in neighbour_lists.items():
            if any(stereo != 0 for _, _, stereo in neighbours) and \
                    any(order == 2 for _, order, _ in neighbours):
                starts = [(neighbour, stereo) for neighbour, order, stereo in neighbours if order == 1]

                if nodes[node_map[i]]["element"] == "C":
                    if len(starts) != 2: continue
                elif nodes[node_map[i]]["element"] == "N":
                    if len(starts) != 1: continue
                    starts.append((-1, 0))
                else:
                    continue
                starts.sort(key=lambda item: item[1])

                internal = [i]
                end_double = [neighbour for neighbour, order, stereo in neighbours if order == 2][0]

                while len(neighbour_lists[end_double]) == 2 and \
                        all(order == 2 for _, order, _ in neighbour_lists[end_double]):
                    internal.append(end_double)
                    end_double = [neighbour for neighbour, order, stereo in neighbour_lists[end_double]
                                  if neighbour != internal[-2]][0]
                internal.append(end_double)

                ends = [(neighbour, stereo) for neighbour, order, stereo in neighbour_lists[end_double] if order == 1]

                if nodes[node_map[end_double]]["element"] == "C":
                    if len(ends) != 2: continue
                elif nodes[node_map[end_double]]["element"] == "N":
                    if len(ends) != 1: continue
                    ends.append((-1, 0))
                else:
                    continue

                ends.sort(key=lambda item: item[1])
                if starts[0][1] == starts[1][1]:
                    raise ValueError("Invalid stereochemistry: " + input)
                if ends[0][1] == ends[1][1]:
                    if ends[0][1] == 0:
                        continue
                    raise ValueError("Invalid stereochemistry: " + input)

                if 0 <= starts[0][0] < ends[0][0] >= 0:
                    stereo_paths.append((starts[0][0], *internal, ends[0][0]))
                if 0 <= starts[1][0] < ends[1][0] >= 0:
                    stereo_paths.append((starts[1][0], *internal, ends[1][0]))

    adjacency_matrix = [[0] * len(nodes) for _ in nodes]
    edges: list[(int, int)] = []

    if parse_cis_trans and apply_paths:
        for path in stereo_paths:
            start = path[0]
            end = path[-1]
            if any(x == end for x, _, _ in neighbour_lists[start]):
                raise ValueError("Invalid stereochemistry: " + input)
            neighbour_lists[start].append((end, -1, 0))
            neighbour_lists[end].append((start, -1, 0))

    for i, neighbours in neighbour_lists.items():
        neighbours.sort(key=lambda item: item[0])
        for j, (neighbour, order, stereo) in enumerate(neighbours):
            adjacency_matrix[node_map[i]][node_map[neighbour]] = 1
            edges.append((order, stereo))

    return [{
        "type": lookup_atom_type(node["element"], is_aromatic=node["aromatic"], unknown_is_dummy=unknown_atom_is_dummy),
    } for node in nodes], adjacency_matrix, [{
        "type": lookup_bond_type(order),
    } for order, _ in edges], stereo_paths


test_inputs = """
N=C(N)NN=C(/C=C\c1ccc([N+](=O)[O-])o1)/C=C\c1ccc([N+](=O)[O-])o1
[As]
[Al+3].[Al+3].[Mo].[Mo].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2]
C[n+]1c(/C=C\c2ccc(O)c3ncccc23)[se]c2ccccc21
N#N
CN=C=O
Cu2+SO2−
[Cu+2].[O-]S(=O)(=O)[O-]
O=Cc1ccc(O)c(OC)c1
COc1cc(C=O)ccc1O
CC(=O)NCCC1=CNc2c1cc(OC)cc2
CC(=O)NCCc1c[nH]c2ccc(OC)cc12
CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4
CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1
CN1CCC[C@H]1c2cccnc2
CCC[C@@H](O)CC\\C=C\\C=C\\C#CC#C\\C=C\\CO
CCC[C@@H](O)CC/C=C/C=C/C#CC#C/C=C/CO
CC1=C(C(=O)C[C@@H]1OC(=O)[C@@H]2[C@H](C2(C)C)/C=C(\\C)/C(=O)OC)C/C=C\\C=C
O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5
OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)1
OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2
CC(=O)OCCC(/C)=C\\C[C@H](C(C)=C)CCC=C
CC[C@H](O1)CC[C@@]12CCCO2
CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2
OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N
CC(C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO
CC
O=C=O
C#N
CCN(CC)CC
CC(=O)O
C1CCCCC1
c1ccccc1
[OH3+]
[2H]O[2H]
F/C=C/F
F/C=C\\F
N[C@@H](C)C(=O)O
N[C@H](C)C(=O)O
[I-].[Na+].C=CCBr
[Na+].[Br-].C=CCI
(C(=O)O).(OCC)
(C(=O)OCC).(O)
OCC	CCO
[CH3][CH2][OH]
CCO
C-C-O
CCO
C(O)C
CCO
OC(=O)C(Br)(Cl)N
ClC(Br)(N)C(=O)O
O=C(O)C(N)(Br)Cl
NC(Cl)(Br)C(=O)O
C
P
N
S
O
Cl
[S]
[H+]
[Fe+2]
[OH-]
[Fe++]
[OH3+]
[NH4+]
CC
C=O
C=C
O=C=O
COC
C#N
CCO
[H][H]
C=CCC=CCO
C=C-C-C=C-C-O
OCC=CCC=C
CCN(CC)CC
CC(C)C(=O)O
C=CC(CCC)C(C(C)C)CCC
C12C3C4C1C5C4C3C25
O1CCCCC1N1CCCCC1
C1.C1
[12C]
[13C]
[C]
[13CH4]
F/C=C/C=C/C
F/C=C/C=CC
NC(C)(F)C(=O)O
NC(F)(C)C(=O)O
N[C@](C)(F)C(=O)O
N[C@@](F)(C)C(=O)O
N[C@@]([H])(C)C(=O)O
N[C@]([H])(C)C(=O)O
N[C@@H](C)C(=O)O
N[C@H](C)C(=O)O
N[C@H](C(=O)O)C
N[C@@H](C(=O)O)C
[H][C@](N)(C)C(=O)O
[H][C@@](N)(C)C(=O)O
[C@H](N)(C)C(=O)O
[C@@H](N)(C)C(=O)O
C[C@H]1CCCCO1
O1CCCC[C@@H]1C
C1=COC=C1
C1=CN=C[NH]C(=O)1
C1=CC=CC=C1
c1cocc1
c1cnc[nH]c(=O)1
c1ccccc1
n1ccccc1
O=n1ccccc1
[O-][n+]1ccccc1
Cn1cccc1
[nH]1cccc1
"""

if __name__ == '__main__':
    for x, i in enumerate(items := test_inputs.split("\n")):
        i = i.strip()
        if i:
            parse_smiles(i, only_biggest=True, unknown_atom_is_dummy=True)
        print(x, "/", len(items), i)
