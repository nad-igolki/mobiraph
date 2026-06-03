from scripts.n07_kmer_fast_calc import process_k
from scripts.n20_kmer_data import DataRepo, load_ids
from scripts.n21_kmer_experiments import (
    prepare_dataset,
    train_model,
    save_model_artifacts,
    evaluate_model,
    root_to_dirname,
)
from pathlib import Path
import pickle
import pandas as pd
from tensorflow import keras


def train_4mer(fasta_path: str, checkpoint_dir: str, metadata_file: str, length: int, processes: int | None = None):
    start_end_fasta = Path(checkpoint_dir) / f"4_{length}_start_end.fasta"

    with open(fasta_path) as f, open(str(start_end_fasta), "w") as out:
        seq_id = None
        seq = []

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    sequence = "".join(seq)

                    start = sequence[:length]
                    end = sequence[-length:]

                    out.write(f">{seq_id}_start\n{start}\n")
                    out.write(f">{seq_id}_end\n{end}\n")

                seq_id = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)

        if seq_id is not None:
            sequence = "".join(seq)

            start = sequence[:length]
            end = sequence[-length:]

            out.write(f">{seq_id}_start\n{start}\n")
            out.write(f">{seq_id}_end\n{end}\n")

    process_k(4, str(start_end_fasta), checkpoint_dir, processes=processes)

    unmerged_csv = Path(checkpoint_dir) / f"4.csv"
    EMBEDDINGS_PATH = Path(checkpoint_dir) / f"4_{length}.csv"

    df = pd.read_csv(unmerged_csv)

    df["seq_id"] = df["name"].str.replace(r"_(start|end)$", "", regex=True)
    df["part"] = df["name"].str.extract(r"_(start|end)$")

    embedding_cols = [c for c in df.columns if c.startswith("emb_")]

    start_df = df[df["part"] == "start"][["seq_id"] + embedding_cols].copy()
    end_df = df[df["part"] == "end"][["seq_id"] + embedding_cols].copy()

    start_df = start_df.rename(columns={c: c for c in embedding_cols})

    end_df = end_df.rename(
        columns={
            c: f"emb_{int(c.split('_')[1]) + 256}"
            for c in embedding_cols
        }
    )

    merged = pd.merge(start_df, end_df, on="seq_id", how="inner")
    merged.rename(columns={'seq_id': 'name'}, inplace=True)

    merged.to_csv(EMBEDDINGS_PATH, index=False)

    METADATA_PATH = metadata_file

    OUTPUTS_DIR = Path(checkpoint_dir) / f"4mer{length}" / "models"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUTS_IMAGES_DIR = Path(checkpoint_dir) / f"4mer{length}"
    OUTPUTS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_LOGITS_RESULTS_DIR = Path(checkpoint_dir) / f"4mer{length}" / "logits"
    OUTPUTS_LOGITS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    EPOCHS = 50

    repo = DataRepo(
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
    )

    all_ids = repo.emb_matrix_df.index.tolist()

    train_ids = all_ids.copy()
    test_ids = all_ids.copy()

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

        print("Classes:", len(label_encoder.classes_))
        print(list(label_encoder.classes_))

        print(f"START: root={repr(hierarchy_root)}, epochs={EPOCHS}")

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
                Path(OUTPUTS_LOGITS_RESULTS_DIR)
                / root_to_dirname(hierarchy_root)
        )
        logit_dir.mkdir(parents=True, exist_ok=True)

        model_path = exp_dir / "cnn_model.keras"
        if model_path.exists():
            print(f"SKIP: already exists -> {model_path}")
            continue

        model, history = train_model(
            X=X_train_full,
            y=y_train_full,
            epochs=20,
            batch_size=BATCH_SIZE,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            best_model_path=f"{exp_dir}/cnn_model.keras"
        )

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
            sample_ids=test_names,
            label_encoder=label_encoder,
            output_dir=img_dir,
            logit_dir=logit_dir,
        )

        print(report_text)
        print(f"SAVED TO: {img_dir} and {logit_dir}")

        print("DONE")


def test_4mer(fasta_path: str, checkpoint_dir: str, models_path: str, length: int, processes: int | None = None):
    start_end_fasta = Path(checkpoint_dir) / f"4_{length}_start_end.fasta"

    with open(fasta_path) as f, open(str(start_end_fasta), "w") as out:
        seq_id = None
        seq = []

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    sequence = "".join(seq)

                    start = sequence[:length]
                    end = sequence[-length:]

                    out.write(f">{seq_id}_start\n{start}\n")
                    out.write(f">{seq_id}_end\n{end}\n")

                seq_id = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)

        if seq_id is not None:
            sequence = "".join(seq)

            start = sequence[:length]
            end = sequence[-length:]

            out.write(f">{seq_id}_start\n{start}\n")
            out.write(f">{seq_id}_end\n{end}\n")

    process_k(4, str(start_end_fasta), checkpoint_dir, processes=processes)

    unmerged_csv = Path(checkpoint_dir) / f"4.csv"
    EMBEDDINGS_PATH = Path(checkpoint_dir) / f"4_{length}.csv"

    df = pd.read_csv(unmerged_csv)

    df["seq_id"] = df["name"].str.replace(r"_(start|end)$", "", regex=True)
    df["part"] = df["name"].str.extract(r"_(start|end)$")

    embedding_cols = [c for c in df.columns if c.startswith("emb_")]

    start_df = df[df["part"] == "start"][["seq_id"] + embedding_cols].copy()
    end_df = df[df["part"] == "end"][["seq_id"] + embedding_cols].copy()

    start_df = start_df.rename(columns={c: c for c in embedding_cols})

    end_df = end_df.rename(
        columns={
            c: f"emb_{int(c.split('_')[1]) + 256}"
            for c in embedding_cols
        }
    )

    merged = pd.merge(start_df, end_df, on="seq_id", how="inner")
    merged.rename(columns={'seq_id': 'name'}, inplace=True)

    merged.to_csv(EMBEDDINGS_PATH, index=False)
    METADATA_PATH = None
    TRAIN_IDS_PATH = None

    OUTPUTS_DIR = Path(models_path) / f"4mer{length}" / "models"

    OUTPUTS_LOGITS_RESULTS_DIR = Path(checkpoint_dir) / f"4mer{length}" / "logits"
    OUTPUTS_LOGITS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    HIERARCHY_ROOTS = [
        "",
        "Class I (Retrotransposons)",
        "Class II (DNA transposons)",
        "Class I (Retrotransposons)\tLTR Retrotransposon",
        "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
    ]

    BATCH_SIZE = 32

    repo = DataRepo(
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
    )

    sample_ids = load_ids(TRAIN_IDS_PATH) if TRAIN_IDS_PATH else None
    has_labels = METADATA_PATH is not None

    for HIERARCHY_ROOT in HIERARCHY_ROOTS:
        exp_dir = Path(OUTPUTS_DIR) / root_to_dirname(HIERARCHY_ROOT)
        out_dir = Path(OUTPUTS_LOGITS_RESULTS_DIR) / root_to_dirname(HIERARCHY_ROOT)
        out_dir.mkdir(parents=True, exist_ok=True)

        model = keras.models.load_model(exp_dir / "cnn_model.keras")

        with open(exp_dir / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        X, y, names = prepare_dataset(
            repo=repo,
            hierarchy_root=HIERARCHY_ROOT,
            sample_ids=sample_ids,
            label_encoder=label_encoder if has_labels else None,
        )

        logits = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
        y_pred_idx = logits.argmax(axis=1)

        df = pd.DataFrame(logits, columns=label_encoder.classes_)
        df.insert(0, "name", names)
        df["y_pred"] = label_encoder.inverse_transform(y_pred_idx)

        if has_labels and y is not None:
            df["y_true"] = label_encoder.inverse_transform(y)

        df.to_csv(out_dir / f"kmer_cnn_4_{length}.csv", index=False)