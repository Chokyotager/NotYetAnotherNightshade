import joblib

clf = joblib.load("extmodel4-3M-models/Solubility_AqSolDB.joblib")

results = clf.predict([[0] * 64])

print(results)
