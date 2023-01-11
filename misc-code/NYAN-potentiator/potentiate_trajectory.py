import os
import random
import math

import json
import joblib

model_directory = "extmodel5hk-models"
iterations = 10000
step_std = 0.1

# Metropolis-Hastings
beta = 1
annealing_alpha = 0.98
step_alpha = 1

preferences = json.loads(open("model_prefs.json").read())

latent = [0] * 64
latent = [x + random.gauss(0, 1.5) for x in latent]

# Nc1nc2c(ncn2[C@H]2CC(O)[C@@H](CO)O2)c(=S)[nH]1
#latent = [-0.8735622763633728, 1.1837552785873413, -1.8812977075576782, -3.2885849475860596, -0.7057079672813416, -0.8465297818183899, 0.8819467425346375, -0.06253132224082947, 0.7778189778327942, -1.740811824798584, 0.41692420840263367, 0.37842807173728943, 0.8186056613922119, -1.5941076278686523, 0.20946815609931946, 1.4517040252685547, -0.2641214430332184, -0.5361548066139221, -1.4120337963104248, -0.08034874498844147, 1.4673113822937012, -0.7070285677909851, -0.6809179186820984, -0.5042502284049988, 0.16180309653282166, -1.0017104148864746, 0.4412194490432739, 0.36681511998176575, -0.2010912001132965, 0.06083561107516289, 1.0268362760543823, -1.1491625308990479, 0.2900262176990509, 1.23662531375885, 0.4088939428329468, 2.0281636714935303, 0.9501795768737793, 1.0582863092422485, -1.3313990831375122, -0.7433129549026489, 2.2754650115966797, -2.089230537414551, -1.9467331171035767, 1.0704576969146729, 0.03805890679359436, 1.737514853477478, 0.24161139130592346, -0.7674322724342346, 0.6030535101890564, -0.296884149312973, -1.3961172103881836, 0.2700660824775696, -0.053815215826034546, -1.1939481496810913, -0.3038332164287567, 0.06440630555152893, 0.359627366065979, 0.5595753192901611, -0.48123010993003845, 0.38839027285575867, -0.3464265763759613, 0.10267198830842972, -0.20925122499465942, 1.810167670249939]

trajectory = list()

#latent = [0] * 64
#latent = [x + random.gauss(0, 1.5) for x in latent]

previous_score = float("-inf")
scale_factor = float()

# Load all models into memory
for model_pref in preferences:

    model_name = model_pref["model"]
    type = model_pref["type"]

    model_pref["clf"] = joblib.load(model_directory + "/" + model_name + ".joblib")

    if type == "classification":

        for label_weight in model_pref["label_weights"]:
            scale_factor += abs(label_weight)

    elif type == "regression":
        scale_factor += abs(model_pref["weight"])

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

                score += result[j] * model_pref["label_weights"][j] / scale_factor

        elif type == "regression":

            result = model_pref["clf"].predict([latent])[0]
            model_scores[model_name] = result.tolist()

            score += result * model_pref["weight"] / scale_factor

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
