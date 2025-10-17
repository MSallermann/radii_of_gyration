import json

accessions = ["P37840", "Q01718", "Q5A5Q6", "P06971", "P13468"]
temperatures = [293, 278, 298, 298, 278, 288]
ionic_strengths = [185, 200, 150, 150, 200, 150]
thresholds = [50, 60, 70, 80, 90]

samples = {}
for idx_acc, acc in enumerate(accessions):
    for thresh in thresholds:
        key = f"{acc}_{thresh}"

        temp = temperatures[idx_acc]
        ionic_strength = ionic_strengths[idx_acc]

        samples[key] = {
            "accession": acc,
            "temperature": temp,
            "ionic_strength": ionic_strength,
            "threshold": thresh,
        }

with open("./samples.json", "w") as f:
    json.dump(samples, f)
