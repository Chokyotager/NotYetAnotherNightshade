import os
import random
import math

import json
import joblib

model_directory = "extmodel5hk-models"
iterations = 10000
step_std = 0.3

# Metropolis-Hastings
beta = 1e256
annealing_alpha = 0.995
step_alpha = 1

preferences = [

    {"model": "Carcinogens_Lagunin", "type": "classification", "weight": [0, 5], "cutoff": [-1, 1], "offset": [0, 0], "further-optimisation": [False, False]}

]

# N=C(N)NCCCCNC(=N)N
latent = [-3.126589775085449, 1.322587490081787, -0.8194237351417542, -0.7977650761604309, -0.6334847807884216, -0.1890188455581665, -2.153082847595215, -1.4730247259140015, -1.72092866897583, -1.3883962631225586, 0.5338287353515625, -1.7137068510055542, -0.4015475809574127, -2.777214527130127, -0.29177457094192505, 0.9661456346511841, -0.8827917575836182, -1.503867745399475, -0.2979474663734436, -0.2995292544364929, -0.15188339352607727, -2.846801996231079, 0.25787296891212463, -2.047144651412964, 1.1977952718734741, -1.55995512008667, -0.30821558833122253, -0.201193705201149, 0.8760308027267456, -0.24222944676876068, 2.0198161602020264, 0.4536007046699524, -0.8948464393615723, -1.2001583576202393, 1.197289228439331, 1.009365200996399, -2.0696725845336914, 1.0456182956695557, -2.0552260875701904, -1.7294158935546875, 1.025734305381775, 0.7269614934921265, -1.966162919998169, -0.26312166452407837, 0.20881162583827972, 0.5561976432800293, -0.4740087389945984, -2.016232490539551, -0.8638930916786194, -1.5499107837677002, -0.41606348752975464, -1.2882342338562012, 2.421170711517334, -0.49234747886657715, 1.0198566913604736, -0.18661430478096008, 0.5422489047050476, -3.5164527893066406, 0.8814524412155151, 1.6787950992584229, -0.0838351845741272, -1.2894225120544434, 1.2915023565292358, 0.6897063255310059]

trajectory = list()

#latent = [0] * 64
#latent = [x + random.gauss(0, 1.5) for x in latent]

previous_score = float("-inf")

def cost_function (x, cutoff, offset=0, allow_further_opt=True):

    if cutoff == 0:
        raise Exception("Cutoff cannot be zero")

    intersect = math.pow(10, -1.5) * cutoff

    if cutoff > 0 and (x - offset) < intersect or cutoff < 0 and (x - offset) > intersect:
        return -(-cutoff/abs(cutoff) * x + cutoff/abs(cutoff) * offset + (math.pow(10, -1.5) * abs(cutoff)) + 1.5)

    elif allow_further_opt:
        return -min(1.5, -math.log10((x - offset) / cutoff))

    else:
        return -max(0, min(1.5, -math.log10((x - offset)/ cutoff)))

# Load all models into memory
for model_pref in preferences:

    model_name = model_pref["model"]
    type = model_pref["type"]

    model_pref["clf"] = joblib.load(model_directory + "/" + model_name + ".joblib")

annealing_temp = 1
step_factor = 1

for i in range(iterations):

    annealing_temp *= annealing_alpha
    step_factor *= step_alpha

    score = float()
    model_scores = dict()

    for model_pref in preferences:

        model_name = model_pref["model"]
        type = model_pref["type"]

        if type == "classification":

            result = model_pref["clf"].predict_proba([latent])[0]
            model_scores[model_name] = result.tolist()

            for j in range(len(result)):

                score += cost_function(result[j], model_pref["cutoff"][j], offset=model_pref["offset"][j], allow_further_opt=model_pref["further-optimisation"][j]) * model_pref["weight"][j]

        elif type == "regression":

            result = model_pref["clf"].predict([latent])[0]
            model_scores[model_name] = result.tolist()

            score += cost_function(result, model_pref["cutoff"], offset=model_pref["offset"], allow_further_opt=model_pref["further-optimisation"]) * model_pref["weight"]

    status = None

    if score > previous_score:

        previous_latent = latent
        previous_score = score

        status = "Accepted"

        trajectory.append({"iteration": i + 1, "score": score, "latent": latent, "model_scores": model_scores})

    else:

        delta = score - previous_score
        accept_probability = math.exp(beta * delta / annealing_temp)

        if random.random() < accept_probability:

            previous_latent = latent
            previous_score = score

            status = "Accepted"

            trajectory.append({"iteration": i + 1, "score": score, "latent": latent, "model_scores": model_scores})

        else:

            latent = previous_latent

            status = "Declined"

    # Randomly move the points
    latent = [x + random.gauss(0, step_std) for x in latent]

    print(i, score, status, "TEMPERATURE:", annealing_temp)

print("\nFinal results")
print("===============")

for model_pref in preferences:

    model_name = model_pref["model"]
    type = model_pref["type"]

    if type == "classification":

        result = model_pref["clf"].predict_proba([latent])[0]
        print(model_name, result)

    elif type == "regression":

        result = model_pref["clf"].predict([latent])[0]
        print(model_name, result)

print("===============")

open("carcinogen_trajectory.json", "w+").write(json.dumps(trajectory, indent=4))
