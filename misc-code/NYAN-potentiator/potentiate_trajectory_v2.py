import os
import random
import math

import json
import joblib

model_directory = "extmodel5hk-models"
iterations = 10000
step_std = 0.01

# Metropolis-Hastings
beta = 1
annealing_alpha = 0.99
step_alpha = 1

preferences = json.loads(open("model_prefs_v2.json").read())

latent = [0] * 64
latent = [x + random.gauss(0, 0.5) for x in latent]

trajectory = list()

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

open("trajectory.json", "w+").write(json.dumps(trajectory, indent=4))
open("latents.json", "w+").write(json.dumps(latent))
