"""Export the trained baseline CNN to an inference-only ONNX file.

This is an optional training/export utility. It is not needed to run the ONNX
model or the portable executable.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import tf2onnx
from tensorflow import keras


MODEL_PATH = os.path.join("models_syntagrus", "cnn_single.keras")
OUTPUT_PATH = "pos_tagging_cnn.onnx"


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Не найдена обученная модель {MODEL_PATH}. Сначала выполните ноутбук."
        )
    model = keras.models.load_model(MODEL_PATH, compile=False)
    signature = (tf.TensorSpec((None, 40, 20), tf.int32, name="characters"),)
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=signature,
        opset=17,
        output_path=OUTPUT_PATH,
    )
    print(f"Сохранена {OUTPUT_PATH}; узлов графа: {len(onnx_model.graph.node)}")


if __name__ == "__main__":
    main()
