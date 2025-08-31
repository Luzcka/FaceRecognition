# db/database.py

import pickle
from typing import List, Dict
from pathlib import Path
import numpy as np

DB_FILE = Path("data/embeddings.pkl")

def save_embedding(name: str, registration_number: str, embedding: np.ndarray):
    """
    Salva embedding facial no arquivo.
    """
    data = load_all_embeddings()
    data.append({"name": name, "registration_number": registration_number, "embedding": embedding})
    with open(DB_FILE, "wb") as f:
        pickle.dump(data, f)

def load_all_embeddings() -> List[Dict]:
    """
    Carrega todos os embeddings faciais salvos.
    """
    if DB_FILE.exists():
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return []
