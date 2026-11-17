"""Быстрый запуск сохранённых POS-моделей без обучения и чтения корпуса.

Примеры:
    python predict_pos.py --text "Мама читает книгу."
    python predict_pos.py --text "Что делать?" --models-dir models_syntagrus

Скрипт использует vocab_syntagrus.json и сохранённые модели из models_syntagrus.
Сначала нужно один раз выполнить обучающий ноутбук pos_tagging_syntagrus.ipynb.
"""

import argparse
import json
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from tensorflow import keras


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_RE = re.compile(r"\w+(?:-\w+)*|[^\w\s]", re.UNICODE)


def load_pos_models(models_dir=None):
    """Загружает только уже сохранённые модели; обучение не запускается."""
    models_dir = models_dir or os.path.join(BASE_DIR, "models_syntagrus")
    vocab_path = os.path.join(BASE_DIR, "vocab_syntagrus.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Не найден {vocab_path}. Сначала выполните обучающий ноутбук."
        )
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    models = {}
    filenames = {
        "CNN": "cnn_single.keras",
        "LSTM": "lstm_single.keras",
        "GRU": "gru_single.keras",
        "Многозадачная LSTM": "lstm_multi.keras",
    }
    for name, filename in filenames.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            models[name] = keras.models.load_model(path, compile=False)
    if not models:
        raise FileNotFoundError(
            f"В {models_dir} нет сохранённых моделей. Выполните ноутбук один раз."
        )
    return models, vocab


def _encode(tokens, vocab):
    max_words = int(vocab.get("max_words", 40))
    max_chars = int(vocab.get("max_chars", 20))
    char2id = vocab["char2id"]
    x = np.zeros((1, max_words, max_chars), dtype="int32")
    for j, word in enumerate(tokens[:max_words]):
        for k, char in enumerate(word[:max_chars]):
            x[0, j, k] = char2id.get(char, 1)
    return x


def _upos_probabilities(model, x):
    result = model.predict(x, verbose=0)
    if isinstance(result, dict):
        return result["upos"], result
    if isinstance(result, list):
        return result[0], dict(zip(model.output_names, result))
    return result, {"upos": result}


def tag_text(text, models_dir=None):
    """Возвращает DataFrame с частями речи и уверенностью моделей."""
    models, vocab = load_pos_models(models_dir)
    upos = vocab["upos"]
    tokens = TOKEN_RE.findall(text)
    max_words = int(vocab.get("max_words", 40))
    rows = []

    for start in range(0, len(tokens), max_words):
        chunk = tokens[start : start + max_words]
        x = _encode(chunk, vocab)
        probabilities = {}
        multi_outputs = None
        for name, model in models.items():
            probs, outputs = _upos_probabilities(model, x)
            probabilities[name] = probs[0]
            if "Многозадачная" in name:
                multi_outputs = outputs

        for j, word in enumerate(chunk):
            row = {"слово": word}
            for name, probs in probabilities.items():
                index = int(probs[j].argmax())
                row[name] = upos[index]
                row[f"уверенность {name}"] = round(float(probs[j].max()), 3)

            if multi_outputs is not None:
                features = []
                values = vocab.get("morph_values", {})
                for key in vocab.get("morph_keys", []):
                    output_name = "morph_" + key
                    if output_name not in multi_outputs:
                        continue
                    probs = multi_outputs[output_name][0][j]
                    if probs.max() >= 0.5:
                        reverse = {int(i): value for value, i in values[key].items()}
                        features.append(f"{key}={reverse[int(probs.argmax())]}")
                row["признаки (многозадачная модель)"] = "|".join(features)
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POS-разбор сохранёнными моделями")
    parser.add_argument("--text", required=True, help="слово или предложение")
    parser.add_argument("--models-dir", default=None, help="папка с *.keras")
    args = parser.parse_args()
    print(tag_text(args.text, args.models_dir).to_string(index=False))
