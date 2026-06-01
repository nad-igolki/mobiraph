bad_sf = ["Academ", "DNA transposon_other", "Kolobok", "Troyka", "Non-LTR Retrotransposon_other", "piggyBac"]

import json

with open('/Users/nad/mobiraph/data/n13_repbase_processed/metadata_03.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

names_to_delete = []
for key, elem_dict in metadata.items():
    if 'superfamily' in elem_dict:
        if elem_dict['superfamily'] in bad_sf:
            names_to_delete.append(key)


with open('/Users/nad/mobiraph/data/n13_repbase_processed/id_test.txt', 'r', encoding='utf-8') as fin, \
     open('/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/id_test.txt', 'w', encoding='utf-8') as fout:

    for i, line in enumerate(fin):
        if line[:-1] not in names_to_delete:
            if line[:-1] in metadata:
                fout.write(f"{line}")



with open('/Users/nad/mobiraph/data/n13_repbase_processed/id_train.txt', 'r', encoding='utf-8') as fin, \
     open('/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/id_train.txt', 'w', encoding='utf-8') as fout:

    for i, line in enumerate(fin):
        if line[:-1] not in names_to_delete:
            if line[:-1] in metadata:
                fout.write(f"{line}")





names_to_delete = []
for key, elem_dict in metadata.items():
    if 'superfamily' not in elem_dict:
        names_to_delete.append(key)


with open('/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/id_test.txt', 'r', encoding='utf-8') as fin, \
     open('/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/id_test_with_superfamilies.txt', 'w', encoding='utf-8') as fout:

    for i, line in enumerate(fin):
        if line[:-1] not in names_to_delete:
            fout.write(f"{line}")



with open('/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/id_train.txt', 'r', encoding='utf-8') as fin, \
     open('/Users/nad/mobiraph/data/n13_repbase_processed_wo_bad_sf/id_train_with_superfamilies.txt', 'w', encoding='utf-8') as fout:

    for i, line in enumerate(fin):
        if line[:-1] not in names_to_delete:
            fout.write(f"{line}")