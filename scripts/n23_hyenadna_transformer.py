import io
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import os

import config

name_to_embedding = {}
name_to_type = {}

path_to_hyena_embedding = f'{config.DIR_HYENA}/hyena_embeddings_1024_and_types.pkl'

def load_pickle_torch_cpu(path):
    orig = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        with open(path, 'rb') as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = orig


def root_to_dirname(root: str) -> str:
    if root == "":
        return "root"
    return root

data = load_pickle_torch_cpu(path_to_hyena_embedding)
name_to_embedding = data['embeddings']

HIERARCHY_ROOTS = [
    "",
    "Class I (Retrotransposons)",
    "Class II (DNA transposons)",
    "Class I (Retrotransposons)\tLTR Retrotransposon",
    "Class I (Retrotransposons)\tNon-LTR Retrotransposon",
]

# Transformer classifier
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch, input_dim]
        x = self.input_proj(x)          # [batch, d_model]
        x = x.unsqueeze(1)              # [batch, 1, d_model]
        x = self.transformer(x)         # [batch, 1, d_model]
        x = x[:, 0, :]                  # [batch, d_model]
        x = self.norm(x)
        logits = self.classifier(x)
        return logits


with open(f'{config.DIR_REPBASE_PROCESSED}/id_train.txt', 'r', encoding='utf-8') as file:
    names_train = [line.strip() for line in file]

with open(f'{config.DIR_REPBASE_PROCESSED}/id_test.txt', 'r', encoding='utf-8') as file:
    names_test = [line.strip() for line in file]

train_filtered = [name for name in names_train if name in name_to_embedding]
test_filtered = [name for name in names_test if name in name_to_embedding]

with open(f'{config.DIR_REPBASE_PROCESSED}/hierarchy_sequences_02_ltr_correction_with_classes.json', 'r', encoding='utf-8') as file:
    meta = json.load(file)

for HIERARCHY_ROOT in HIERARCHY_ROOTS:
    print('-' * 20, HIERARCHY_ROOT, '-' * 20)

    save_dir = f"/Users/nad/mobiraph/data/n23_hyena_models/{root_to_dirname(HIERARCHY_ROOT)}"
    model_path = os.path.join(save_dir, "transformer_model.pt")

    if os.path.exists(model_path):
        print(f"Модель уже существует: {model_path}. Пропускаю обучение.")
        continue

    current_meta = meta.copy()
    for part in HIERARCHY_ROOT.split("\t"):
        if not part:
            continue
        current_meta = current_meta[part]["subs"]
    name_to_type = {}
    for class_name, class_dict in current_meta.items():
        for seq in class_dict["sequences"]:
            name_to_type[seq] = class_name

    with open(f'{config.DIR_REPBASE_PROCESSED}/id_train.txt', 'r', encoding='utf-8') as file:
        names_train = [line.strip() for line in file]

    with open(f'{config.DIR_REPBASE_PROCESSED}/id_test.txt', 'r', encoding='utf-8') as file:
        names_test = [line.strip() for line in file]

    train_filtered_root = [name for name in train_filtered if name in name_to_type]
    test_filtered_root = [name for name in test_filtered if name in name in name_to_type]

    X_train = np.array([name_to_embedding[name] for name in train_filtered_root], dtype=np.float32)
    y_train = np.array([name_to_type[name] for name in train_filtered_root])

    X_test = np.array([name_to_embedding[name] for name in test_filtered_root], dtype=np.float32)
    y_test = np.array([name_to_type[name] for name in test_filtered_root])

    print('shape X_train', X_train.shape, 'shape y_train', y_train.shape)
    print('shape X_test', X_test.shape, 'shape y_test', y_test.shape)

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Кодирование меток
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    num_classes = len(label_encoder.classes_)
    input_dim = X_train_scaled.shape[1]
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train_enc)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_enc
    )
    class EmbeddingDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = EmbeddingDataset(X_train_scaled, y_train_enc)
    test_dataset = EmbeddingDataset(X_test_scaled, y_test_enc)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dataset
    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Обучение
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, loss={avg_loss:.4f}")

    save_dir = f"/Users/nad/mobiraph/data/n23_hyena_models/{root_to_dirname(HIERARCHY_ROOT)}"
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "label_encoder": label_encoder,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hierarchy_root": HIERARCHY_ROOT,
    }, os.path.join(save_dir, "transformer_model.pt"))

    import pandas as pd
    model.eval()

    all_logits = []
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)

            all_logits.append(logits.cpu())
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y_batch.cpu().numpy())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    class_names = label_encoder.classes_
    all_preds_labels = label_encoder.inverse_transform(all_preds)
    all_true_labels = label_encoder.inverse_transform(all_true)

    df_logits = pd.DataFrame(all_logits, columns=class_names)
    df_logits.insert(0, "name", test_filtered_root)

    df_logits["y_true"] = all_true_labels
    df_logits["y_pred"] = all_preds_labels

    logit_dir = f"/Users/nad/mobiraph/data/n22_test_results/{root_to_dirname(HIERARCHY_ROOT)}"
    os.makedirs(logit_dir, exist_ok=True)

    df_logits.to_csv(f"{logit_dir}/hyena_transformer.csv", index=False)

    print(df_logits.head())

    print("Accuracy:", accuracy_score(all_true_labels, all_preds_labels))
    print("F1-macro:", f1_score(all_true_labels, all_preds_labels, average="macro"))