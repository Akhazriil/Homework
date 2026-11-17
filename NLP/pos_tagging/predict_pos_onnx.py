"""Лёгкий inference POS-модели без TensorFlow/Keras.

Запуск:
    python predict_pos_onnx.py --text "Что делать, если пароль забыт?"

Нужны только NumPy и ONNX Runtime. Файл pos_tagging_cnn.onnx — экспорт лучшей
CNN-модели, выбранной по результатам проверки на nerus_lenta.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "pos_tagging_cnn.onnx"
DEFAULT_VOCAB = HERE / "vocab_syntagrus.json"
TOKEN_RE = re.compile(r"\w+(?:-\w+)*|[^\w\s]", re.UNICODE)


class PosTagger:
    def __init__(self, model_path=DEFAULT_MODEL, vocab_path=DEFAULT_VOCAB):
        model_path = Path(model_path)
        vocab_path = Path(vocab_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Не найдена ONNX-модель: {model_path}")
        if not vocab_path.is_file():
            raise FileNotFoundError(f"Не найден словарь: {vocab_path}")

        with vocab_path.open(encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.char2id = self.vocab["char2id"]
        self.upos = self.vocab["upos"]
        self.max_words = int(self.vocab.get("max_words", 40))
        self.max_chars = int(self.vocab.get("max_chars", 20))

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _encode(self, words):
        x = np.zeros((1, self.max_words, self.max_chars), dtype=np.int32)
        for word_index, word in enumerate(words[: self.max_words]):
            for char_index, char in enumerate(word[: self.max_chars]):
                x[0, word_index, char_index] = self.char2id.get(char, 1)  # 1 = UNK
        return x

    def tag(self, text):
        """Возвращает список слов с предсказанным UPOS и уверенностью CNN."""
        tokens = TOKEN_RE.findall(text)
        result = []
        for start in range(0, len(tokens), self.max_words):
            chunk = tokens[start : start + self.max_words]
            x = self._encode(chunk)
            probabilities = self.session.run(
                [self.output_name], {self.input_name: x}
            )[0][0]
            for word, scores in zip(chunk, probabilities):
                label = int(scores.argmax())
                result.append(
                    {
                        "word": word,
                        "pos": self.upos[label],
                        "confidence": float(scores[label]),
                    }
                )
        return result


def main():
    parser = argparse.ArgumentParser(description="Лёгкая модель POS-разметки русского текста (ONNX)")
    parser.add_argument("--text", required=True, help="Слово или предложение")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Путь к ONNX-модели")
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB), help="Путь к vocab_syntagrus.json")
    args = parser.parse_args()

    tagger = PosTagger(args.model, args.vocab)
    print(f"{'Слово':<20} {'UPOS':<10} Уверенность")
    print("-" * 43)
    for item in tagger.tag(args.text):
        print(f"{item['word']:<20} {item['pos']:<10} {item['confidence']:.3f}")


if __name__ == "__main__":
    main()
