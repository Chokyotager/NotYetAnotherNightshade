from rdkit import Chem
from rdkit.Chem import AllChem

smiles = open("ZINC_centres.smi").read().split("\n")

for smile_entry in smiles:

	current_smiles = smile_entry.split("\t")[0]
	mol = Chem.MolFromSmiles(current_smiles)

	fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=512, useBondTypes=True)

	print(current_smiles + "\t" + fp.ToBitString())
