import config


from scripts.n03_create_dataset import create_dataset
from scripts.n04_kmer_statistics import kmer_distribution


import pandas as pd
from tqdm import tqdm
import os
from Bio import SeqIO


def create_kmer_embeddings(path_to_df: str, k_list: list, mode: int, seed: int | None = None) -> None:
    df = pd.read_csv(path_to_df)
    dataset = create_dataset(df, mode, seed)
    for k in k_list:
        print('Generating kmer embeddings for {}'.format(k))
        dataset_embeddings = pd.DataFrame(columns=["name", "type"])

        for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
            insect_name = row['insect']
            sequence_path = os.path.join(config.DIR_GRAPH_PARTITION, f'{insect_name}_seq_sv_large.fasta')
            rec_sequence = next((r for r in SeqIO.parse(sequence_path, "fasta") if r.id == row['name']), None)
            sequence = str(rec_sequence.seq).upper()
            _, kmer_embedding = kmer_distribution(sequence, k)
            embedding_dict = {
                "name": row['name'],
                "type": row['type'],
                **{f"emb_{i}": val for i, val in enumerate(kmer_embedding)}
            }
            new_row = pd.DataFrame([embedding_dict])
            dataset_embeddings = pd.concat([dataset_embeddings, new_row], ignore_index=True)

        dataset_path = os.path.join(config.DIR_KMER_DATASETS, f"{k}_{mode}_{seed}.csv")
        dataset_embeddings.to_csv(dataset_path, index=False)
        print('Saved dataset to {}'.format(dataset_path), 'Dataset shape: {}'.format(dataset_embeddings.shape), 'Dataset head: {}'.format(dataset_embeddings.head()))


if __name__ == '__main__':
    ks = [4, 5, 6, 7]
    create_kmer_embeddings(config.FILE_UNIQUE_TYPE_PARTITION, ks, mode=0, seed=1)