import csv

neuralte_test = {}

with open("/Users/nad/NeuralTE/classified.info", newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        neuralte_test[row["#Seq_Name"]] = row["Predict_Label"]



import json
with open("/Users/nad/mobiraph/data/n13_repbase_processed/metadata_03.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

true_class = {}

for name in neuralte_test:
    if name in metadata:
        if 'superfamily' in metadata[name]:
            true_class[name] = metadata[name]['superfamily']
    else:
        if name.removesuffix("-intactLTR") + '-LTR' in metadata:
            name = name.removesuffix("-intactLTR")
            if 'superfamily' in metadata[name + '-LTR']:
                true_class[name] = metadata[name + '-LTR']['superfamily']
        elif name.removesuffix("-intactLTR") + '-I' in metadata:
            name = name.removesuffix("-intactLTR")
            if 'superfamily' in metadata[name + '-I']:
                true_class[name] = metadata[name + '-I']['superfamily']
        else:
            print(name)




superfamily_to_class = {
    # Class 2 — DNA transposons
    "Mariner/Tc1": "Class II (DNA transposons)",
    "hAT": "Class II (DNA transposons)",
    "MuDR": "Class II (DNA transposons)",
    "EnSpm/CACTA": "Class II (DNA transposons)",
    "piggyBac": "Class II (DNA transposons)",
    "Harbinger": "Class II (DNA transposons)",
    "Helitron": "Class II (DNA transposons)",
    "Kolobok": "Class II (DNA transposons)",
    "Academ": "Class II (DNA transposons)",
    "DNA transposon_other": "Class II (DNA transposons)",

    # Class 1 — LTR retrotransposons
    "Gypsy": "Class I (Retrotransposons)",
    "Copia": "Class I (Retrotransposons)",
    "BEL": "Class I (Retrotransposons)",
    "DIRS": "Class I (Retrotransposons)",
    "Troyka": "Class I (Retrotransposons)",

    # Class 1 — Non-LTR retrotransposons
    "SINE": "Class I (Retrotransposons)",
    "L1": "Class I (Retrotransposons)",
    "RTE": "Class I (Retrotransposons)",
    "CR1": "Class I (Retrotransposons)",
    "Tx1": "Class I (Retrotransposons)",
    "RTEX": "Class I (Retrotransposons)",
    "Tad1": "Class I (Retrotransposons)",
    "Non-LTR Retrotransposon_other": "Class I (Retrotransposons)"
}

mapping = {
    'Bel-Pao': 'BEL',
    'CACTA': 'EnSpm/CACTA',
    'Copia': 'Copia',
    'Crypton': 'DNA transposon_other',
    'DIRS': 'DIRS',
    'Gypsy': 'Gypsy',
    'Helitron': 'Helitron',
    'I': 'Non-LTR Retrotransposon_other',
    'Jockey': 'Non-LTR Retrotransposon_other',
    'L1': 'L1',
    'Merlin': 'DNA transposon_other',
    'Mutator': 'MuDR',
    'P': 'DNA transposon_other',
    'PIF-Harbinger': 'Harbinger',
    'Penelope': 'Non-LTR Retrotransposon_other',
    'R2': 'Non-LTR Retrotransposon_other',
    'RTE': 'RTE',
    'Retrovirus': 'Gypsy',
    'Tc1-Mariner': 'Mariner/Tc1',
    'Transib': 'DNA transposon_other',
    'Unknown': 'DNA transposon_other',
    'hAT': 'hAT',
    'tRNA': 'SINE'
}


predicted_class = {}

for name in neuralte_test:
    if name in true_class:
        if 'superfamily' in metadata[name]:
        # predicted_class[name] = metadata[name]['class']
            if neuralte_test[name] in mapping:
                predicted_class[name] = mapping[neuralte_test[name]]
            else:
                predicted_class[name] = None






exclude = {
    "Academ",
    "DNA transposon_other",
    "Kolobok",
    "Troyka",
    "Non-LTR Retrotransposon_other",
    "piggyBac"
}

from sklearn.metrics import classification_report

common_keys = set(true_class) & set(predicted_class)

pairs = [
    (true_class[k], predicted_class[k])
    for k in common_keys
    if (
        true_class[k] is not None
        and predicted_class[k] is not None
        and true_class[k] not in exclude
        and predicted_class[k] not in exclude
    )
]

y_true_list = [t for t, _ in pairs]
y_pred_list = [p for _, p in pairs]

print(classification_report(y_true_list, y_pred_list))