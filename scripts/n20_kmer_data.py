#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import pandas as pd


class DataRepo:
    def __init__(self, embeddings_path: str, metadata_path: str):
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)

        self.embeddings_df = pd.read_csv(self.embeddings_path)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.emb_cols = [c for c in self.embeddings_df.columns if c.startswith("emb_")]

        indexed_df = self.embeddings_df.set_index("name")
        indexed_df = indexed_df[~indexed_df.index.duplicated(keep="first")]

        self.emb_matrix_df = indexed_df[self.emb_cols].astype(np.float32)
        self.available_ids = set(self.emb_matrix_df.index)

    def get_meta_node(self, hierarchy_root: str):
        if not hierarchy_root:
            return self.metadata

        current_meta = self.metadata
        for part in hierarchy_root.split("\t"):
            if not part:
                continue
            current_meta = current_meta[part]["subs"]

        return current_meta


def load_ids(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]