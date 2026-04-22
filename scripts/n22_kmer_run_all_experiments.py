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
METADATA_PATH = f"{config.DIR_REPBASE_PROCESSED}/hierarchy_sequences_02_ltr_correction_with_classes.json"
TRAIN_IDS_PATH = f"{config.DIR_REPBASE_PROCESSED}/id_train.txt"
TEST_IDS_PATH = f"{config.DIR_REPBASE_PROCESSED}/id_test.txt"

OUTPUTS_DIR = "/Users/nad/mobiraph/data/n20_kmer_models/ft_transformer"
OUTPUTS_IMAGES_DIR = "/Users/nad/mobiraph/figures/kmer_ft_transformer"
OUTPUTS_TEST_RESULTS_DIR = "/Users/nad/mobiraph/data/n40_tf_transformer_train_results"

HIERARCHY_ROOTS = [
    "",
    "Class I (Retrotransposons)",
    "Class II (DNA transposons)",
    "Class I (Retrotransposons)\tLTR Retrotransposon",
    "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
]

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

        X_train_full, y_train_full, train_names, label_encoder = prepare_dataset(
            repo=repo,
            hierarchy_root=hierarchy_root,
            sample_ids=train_ids,
            label_encoder=None,
        )

        X_test, y_test, test_names = prepare_dataset(
            repo=repo,
            hierarchy_root=hierarchy_root,
            sample_ids=test_ids,
            label_encoder=label_encoder,
        )

        print("Train:", X_train_full.shape, y_train_full.shape)
        print("Test:", X_test.shape, y_test.shape)
        print("Classes:", len(label_encoder.classes_))
        print(list(label_encoder.classes_))

        print(f"START: root={repr(hierarchy_root)}, epochs={50}")

        exp_dir = (
                Path(OUTPUTS_DIR)
                / root_to_dirname(hierarchy_root)
        )
        exp_dir.mkdir(parents=True, exist_ok=True)

        img_dir = (
                Path(OUTPUTS_IMAGES_DIR)
                / root_to_dirname(hierarchy_root)
        )
        img_dir.mkdir(parents=True, exist_ok=True)

        logit_dir = (
                Path(OUTPUTS_TEST_RESULTS_DIR)
                / root_to_dirname(hierarchy_root)
        )
        logit_dir.mkdir(parents=True, exist_ok=True)

        model_path = exp_dir / "best_ft_transformer.keras"
        if model_path.exists():
            print(f"SKIP: already exists -> {model_path}")
            continue

        model, history = train_model(
            X=X_train_full,
            y=y_train_full,
            epochs=2,
            batch_size=BATCH_SIZE,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            best_model_path=f"{exp_dir}/best_ft_transformer.keras"
        )


        save_model_artifacts(
            model=model,
            label_encoder=label_encoder,
            history=history,
            output_dir=exp_dir,
        )
        print(f"SAVED artifacts TO: {exp_dir}")
        print(set(test_names) - set(test_ids))
        print(set(test_ids) - set(test_names))
        report_text = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            sample_ids=test_names,
            label_encoder=label_encoder,
            output_dir=img_dir,
            logit_dir=logit_dir,
        )

        print(report_text)
        print(f"SAVED TO: {img_dir} and {logit_dir}")

    print("DONE")


if __name__ == "__main__":
    main()