#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from scripts.n20_kmer_data import DataRepo, load_ids
from scripts.n21_kmer_experiments import (
    prepare_dataset,
    train_model,
    save_model_artifacts,
    evaluate_model,
    root_to_dirname,
)
import config


EMBEDDINGS_PATH = f"{config.DIR_ALL_SEQUENCES_FILTERED_KMER}/7.csv"
METADATA_PATH = f"{config.DIR_REPBASE_PROCESSED}/hierarchy_sequences_02_ltr_correction.json"
TRAIN_IDS_PATH = f"{config.DIR_REPBASE_PROCESSED}/id_train.txt"
TEST_IDS_PATH = f"{config.DIR_REPBASE_PROCESSED}/id_test.txt"

OUTPUTS_DIR = "/Users/nad/mobiraph/data/n20_kmer_models"
OUTPUTS_IMAGES_DIR = "/Users/nad/mobiraph/figures/kmer_cnn"

HIERARCHY_ROOTS = [
    "",
    "DNA transposon",
    "LTR Retrotransposon",
    "Non-LTR Retrotransposon",
]

EPOCHS_LIST = [10, 20, 30, 40, 50]

BATCH_SIZE = 32
TEST_SIZE = 0.05
RANDOM_STATE = 42


def main():
    repo = DataRepo(
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
    )

    train_ids = load_ids(TRAIN_IDS_PATH)
    test_ids = load_ids(TEST_IDS_PATH)

    for hierarchy_root in HIERARCHY_ROOTS:
        print("=" * 80)
        print("HIERARCHY ROOT:", repr(hierarchy_root))

        X_train_full, y_train_full, _, label_encoder = prepare_dataset(
            repo=repo,
            hierarchy_root=hierarchy_root,
            sample_ids=train_ids,
            label_encoder=None,
        )

        X_test, y_test, _ = prepare_dataset(
            repo=repo,
            hierarchy_root=hierarchy_root,
            sample_ids=test_ids,
            label_encoder=label_encoder,
        )

        print("Train:", X_train_full.shape, y_train_full.shape)
        print("Test:", X_test.shape, y_test.shape)
        print("Classes:", len(label_encoder.classes_))
        print(list(label_encoder.classes_))

        for epochs in EPOCHS_LIST:
            print("-" * 80)
            print(f"START: root={repr(hierarchy_root)}, epochs={epochs}")

            model, history = train_model(
                X=X_train_full,
                y=y_train_full,
                epochs=epochs,
                batch_size=BATCH_SIZE,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

            exp_dir = (
                Path(OUTPUTS_DIR)
                / root_to_dirname(hierarchy_root)
                / f"epochs_{epochs}"
            )
            exp_dir.mkdir(parents=True, exist_ok=True)

            img_dir = (
                    Path(OUTPUTS_IMAGES_DIR)
                    / root_to_dirname(hierarchy_root)
                    / f"epochs_{epochs}"
            )
            img_dir.mkdir(parents=True, exist_ok=True)


            save_model_artifacts(
                model=model,
                label_encoder=label_encoder,
                history=history,
                output_dir=exp_dir,
            )
            print(f"SAVED artifacts TO: {exp_dir}")

            report_text = evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                label_encoder=label_encoder,
                output_dir=img_dir,
            )

            print(report_text)
            print(f"SAVED TO: {img_dir}")

    print("DONE")


if __name__ == "__main__":
    main()