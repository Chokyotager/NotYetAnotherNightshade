import warnings

warnings.filterwarnings("ignore")

datapoints = open("mordred.tsv").read().split("\n")
headers = open("mordred_headers.txt").read().split("\n")
fingerprints = open("morgan512_maccs.tsv").read().split("\n")

for i in range(len(fingerprints)):

    if fingerprints[i] == "":
        continue

    fingerprint = fingerprints[i].split("\t")
    mordred = datapoints[i].split("\t")

    if fingerprint[0] != mordred[0]:
        continue

    print(mordred[0] + "\t" + fingerprint[1] + "," + mordred[1])

exit()
