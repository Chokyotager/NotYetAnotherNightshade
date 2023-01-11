import os
import random
import math

import json
import joblib

model_directory = "extmodel5hk-models"
iterations = 10000
step_std = 0.005

# Metropolis-Hastings
beta = 1
annealing_alpha = 0.99
step_alpha = 1

preferences = json.loads(open("model_prefs.json").read())

# Subsetted over 1000 datapoints; mean, std

#reference_latent = [[-1.4811447066804395, 1.7052989809816517], [-0.6033874022886156, 1.9174140033455151], [-1.5735202315114438, 1.8672604386461167], [-1.3128128921687603, 2.1158410310695626], [1.0027127817496657, 1.6734300412041174], [0.5940955890752375, 1.891961299732078], [2.4097544024810196, 2.9498403983107266], [5.122753319501877, 0.8777288046694934], [0.259595906548202, 1.4631528679251578], [-0.715817446956411, 1.8636032603615158], [-0.5193798680528998, 1.3568908556630341], [-3.5598537365570664, 1.463998223749395], [-0.6118301698286086, 0.9881208862491735], [0.35160920390672984, 2.096472793040274], [0.7705213516317307, 1.2965905813899208], [0.5834951292797923, 2.1539630560613925], [4.619068320035934, 1.2182891090692363], [-0.9802472685948014, 2.0148407139691047], [1.7873976132832468, 1.0102668518931324], [-0.08208747391682118, 1.4029308375895513], [-1.845971031760797, 1.5891126978122883], [6.8055665202140805, 1.9170719905898495], [5.1510873988866805, 1.639113362458199], [-1.1270333957485854, 2.234083250233593], [-3.0769086004793644, 1.122768312728715], [1.3025201110895723, 1.3789858189127837], [2.0277379222512244, 0.7989240568443801], [6.678532440543175, 6.13910139306365], [0.7252676762752235, 2.0773369116199687], [1.7882750063221902, 1.4221213635913663], [1.8027144036712124, 2.319097472852191], [0.3354474254939705, 1.6745302168179772], [0.08378753858804702, 1.9229752719006779], [1.3771632388196886, 1.3155114569859656], [2.188809872340411, 1.1539255466860956], [-3.166528468832374, 1.4385022137760142], [-3.0017398116737604, 1.1826419475393963], [-1.2315909596104175, 1.4965338980971425], [-1.1481668454091996, 1.6477201620834725], [-2.6707336814589797, 2.0453112859978386], [-5.287448148012161, 1.5754323463225157], [1.8623809204883874, 1.7831454599947392], [-2.6116024600937964, 2.186217127906127], [1.3813849059864878, 1.4693013815556546], [-1.523070399732329, 1.9090413654531078], [2.2526235713150817, 3.0602191342693685], [-2.6499902397207915, 1.7313308002983254], [-7.8248930661678315, 1.8027159957226313], [3.0195830558538437, 1.0188336586822404], [3.322622394382954, 1.4377972940385286], [-0.36407162505667656, 2.2606024350873035], [0.11112278135493397, 1.7251506929681362], [0.0615270645711571, 2.2951808585075644], [3.0124843568205835, 1.3564660397069639], [4.848255901828408, 2.51672009729167], [0.8020944022927433, 1.1408687598196694], [-4.402258321329951, 1.5639108201227492], [-8.136487945079804, 1.1820689892073262], [-1.2088985633701086, 1.3603704262051082], [-3.489665960069746, 1.8994513058164435], [-3.0366220222823324, 2.638249396749778], [-7.38560500305891, 2.3793676152806085], [-2.544760019402951, 2.230993699023622], [-0.28705252828635275, 1.3001909840212746]]

#reference_latent = [[0, 0.1]] * 64
#latent = [0 + random.gauss(0, 1) for x in reference_latent]

latent = [0] * 64
latent = [x + random.gauss(0, 1.5) for x in latent]

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

    for model_pref in preferences:

        model_name = model_pref["model"]
        type = model_pref["type"]

        if type == "classification":

            result = model_pref["clf"].predict_proba([latent])[0]

            for j in range(len(result)):

                score += result[j] * model_pref["label_weights"][j] / scale_factor

        elif type == "regression":

            result = model_pref["clf"].predict([latent])[0]
            score += result * model_pref["weight"] / scale_factor

    status = None

    if score > previous_score:

        previous_latent = latent
        previous_score = score

        status = "Accepted"

    else:

        delta = score - previous_score
        accept_probability = math.exp(beta * delta / annealing_temp)

        if random.random() < accept_probability:

            previous_latent = latent
            previous_score = score

            status = "Accepted"

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

open("latents.json", "w+").write(json.dumps(latent))
