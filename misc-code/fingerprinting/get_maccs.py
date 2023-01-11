from rdkit import Chem
from rdkit.Chem import MACCSkeys

smiles = open("ZINC_centres.smi").read().split("\n")

for smile_entry in smiles:

	current_smiles = smile_entry.split("\t")[0]
	mol = Chem.MolFromSmiles(current_smiles)

	fp = MACCSkeys.GenMACCSKeys(mol)

	print(current_smiles + "\t" + fp.ToBitString())
