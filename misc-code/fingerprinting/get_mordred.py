from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors

calc = Calculator(descriptors, ignore_3D=True)
smiles = open("ZINC_centres.smi").read().split("\n")

for smile_entry in smiles:

	current_smiles = smile_entry.split("\t")[0]
	mol = Chem.MolFromSmiles(current_smiles)

	descriptors = calc(mol)

	raw_descriptors = descriptors.values()
	final_descriptors = list()

	for raw_descriptor in raw_descriptors:

		if type(raw_descriptor) is float or type(raw_descriptor) is int:

			final_descriptors.append(str(raw_descriptor))

		else:

			final_descriptors.append("")
            
	print(current_smiles + "\t" + ",".join(final_descriptors))
