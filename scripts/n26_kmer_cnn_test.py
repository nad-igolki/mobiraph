from pathlib import Path
import pickle
import pandas as pd
from tensorflow import keras

from scripts.n20_kmer_data import DataRepo, load_ids
from scripts.n21_kmer_experiments import prepare_dataset, root_to_dirname
import config


EMBEDDINGS_PATH = f"{config.DIR_ALL_SEQUENCES_FILTERED_KMER}/4_5.csv"
# EMBEDDINGS_PATH = f"/Users/nad/mobiraph/data/insects/7.csv"
METADATA_PATH = f"{config.DIR_REPBASE_PROCESSED}/hierarchy_sequences_02_ltr_correction_with_classes.json"
# METADATA_PATH = f"/Users/nad/mobiraph/data/n26_sv_processed/hierarchy_sequences_sv_insects.json"
TRAIN_IDS_PATH = f"{config.DIR_REPBASE_PROCESSED}/id_train.txt"
# TRAIN_IDS_PATH = None

OUTPUTS_DIR = "/Users/nad/mobiraph/data/n32_kmer_models_4_5"
OUTPUTS_RESULTS_DIR = "/Users/nad/mobiraph/data/n34_train_results_new"

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

sample_ids = None
if TRAIN_IDS_PATH:
    sample_ids = load_ids(TRAIN_IDS_PATH)

for HIERARCHY_ROOT in HIERARCHY_ROOTS:
    exp_dir = Path(OUTPUTS_DIR) / root_to_dirname(HIERARCHY_ROOT)
    out_dir = Path(OUTPUTS_RESULTS_DIR) / root_to_dirname(HIERARCHY_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(exp_dir / "cnn_model.keras")

    with open(exp_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    X, y, names = prepare_dataset(
        repo=repo,
        hierarchy_root=HIERARCHY_ROOT,
        sample_ids=sample_ids,
        label_encoder=label_encoder,
    )

    logits = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
    y_pred_idx = logits.argmax(axis=1)

    df = pd.DataFrame(logits, columns=label_encoder.classes_)
    df.insert(0, "name", names)
    df["y_true"] = label_encoder.inverse_transform(y)
    df["y_pred"] = label_encoder.inverse_transform(y_pred_idx)

    df.to_csv(out_dir / "kmer_cnn.csv", index=False)