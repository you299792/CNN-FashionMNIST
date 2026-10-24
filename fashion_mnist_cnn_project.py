import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(r"c:\Users\Acer\Downloads\archive (2)")
OUT_DIR = ROOT / "fashion_mnist_results"
OUT_DIR.mkdir(exist_ok=True)

LABEL_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
INPUT_SIZE = (32, 32)


def load_dataset():
    train_df = pd.read_csv(ROOT / "fashion-mnist_train.csv")
    test_df = pd.read_csv(ROOT / "fashion-mnist_test.csv")
    X_train = train_df.drop(columns=["label"]).astype(np.float32).values.reshape(-1, 28, 28, 1) / 255.0
    y_train = train_df["label"].astype(np.int32).values
    X_test = test_df.drop(columns=["label"]).astype(np.float32).values.reshape(-1, 28, 28, 1) / 255.0
    y_test = test_df["label"].astype(np.int32).values
    return X_train, y_train, X_test, y_test


def resize_images(X):
    X = tf.image.resize(X, size=INPUT_SIZE, method="bilinear").numpy()
    return X.astype(np.float32)


def class_distribution_plot(y_train, y_val, y_test):
    train_counts = pd.Series(y_train).value_counts().sort_index().reindex(range(10), fill_value=0)
    val_counts = pd.Series(y_val).value_counts().sort_index().reindex(range(10), fill_value=0)
    test_counts = pd.Series(y_test).value_counts().sort_index().reindex(range(10), fill_value=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, counts, title in [
        (axes[0], train_counts, "Train set class counts"),
        (axes[1], val_counts, "Validation set class counts"),
        (axes[2], test_counts, "Test set class counts"),
    ]:
        counts.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.set_xticks(range(10))
        ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "task1_class_distribution.png", dpi=200)
    plt.close(fig)


def save_task1_summary(y_train, y_val, y_test):
    counts_train = pd.Series(y_train).value_counts().sort_index().reindex(range(10), fill_value=0)
    counts_val = pd.Series(y_val).value_counts().sort_index().reindex(range(10), fill_value=0)
    counts_test = pd.Series(y_test).value_counts().sort_index().reindex(range(10), fill_value=0)

    with (OUT_DIR / "task1_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Fashion-MNIST data plan\n")
        f.write("=====================\n")
        f.write(f"Task statement: Multi-class clothing image classification using Fashion-MNIST. Input size: {INPUT_SIZE[0]}x{INPUT_SIZE[1]} grayscale images (converted from 28x28 source). Number of classes: 10.\n")
        f.write("Splits: train/validation/test = 48,000 / 6,000 / 10,000 images.\n")
        f.write("Preprocessing: normalize to [0,1], resize from 28x28 to 32x32, and use one-hot labels only at the model output.\n")
        f.write("\nClass counts:\n")
        for i, name in enumerate(LABEL_NAMES):
            total = int(counts_train.iloc[i] + counts_val.iloc[i] + counts_test.iloc[i])
            f.write(f"{i}: {name} -> train={int(counts_train.iloc[i])}, val={int(counts_val.iloc[i])}, test={int(counts_test.iloc[i])}, total={total}\n")


def baseline_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def stronger_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_training(history, model_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["accuracy"], label="Train Accuracy", color="tab:blue")
    ax.plot(history.history["val_accuracy"], label="Val Accuracy", color="tab:orange")
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    ax.axvline(best_epoch - 1, color="red", linestyle="--", label=f"Best val epoch={best_epoch}")
    ax.set_title(f"{model_name} training curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name.lower().replace(' ', '_')}_training.png", dpi=200)
    plt.close(fig)


def evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=4,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=0,
    )

    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    best_val_acc = float(np.max(history.history["val_accuracy"]))
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    plot_training(history, model_name)

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(10))

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{model_name} confusion matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=200)
    plt.close(fig)

    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "history": history,
        "cm": cm,
        "y_pred": y_pred,
    }


def write_report(results):
    table = (
        "| Model | Best Val Acc | Epoch | Test Acc | Notes |\n"
        "|---|---:|---:|---:|---|\n"
        f"| User-Defined Baseline | {results['baseline']['best_val_acc']:.4f} | {results['baseline']['best_epoch']} | {results['baseline']['test_acc']:.4f} | Small CNN with 2 pooling blocks; strong quick baseline. |\n"
        f"| Stronger CNN | {results['stronger']['best_val_acc']:.4f} | {results['stronger']['best_epoch']} | {results['stronger']['test_acc']:.4f} | Deeper stacking with BatchNorm and dropout; higher capacity. |\n"
    )

    cm = results["stronger"]["cm"]
    classes = LABEL_NAMES
    row_sums = cm.sum(axis=1)
    confusion_rows = []
    for i, cls in enumerate(classes):
        row = cm[i]
        top_idx = np.argsort(row)[::-1][:3]
        highlighted = ", ".join(f"{classes[j]} ({row[j]})" for j in top_idx if row[j] > 0)
        confusion_rows.append(f"- {cls}: most common false predictions = {highlighted}; true samples = {row_sums[i]}.")
    confusion_text = "\n".join(confusion_rows)

    report_path = OUT_DIR / "fashion_mnist_report.md"
    report_path.write_text(
        "# Fashion-MNIST CNN report\n\n"
        "## Task 1) Problem framing & data plan\n\n"
        "- Task statement: Multi-class clothing image classification using Fashion-MNIST.\n"
        "- Input size: 32x32 grayscale images after resizing from the source 28x28 format.\n"
        "- Classes: 10 clothing categories.\n"
        "- Splits: train 48,000, validation 6,000, test 10,000.\n"
        "- Preprocessing: normalize to [0,1], resize to 32x32, and keep a fixed random seed for reproducibility.\n\n"
        "## Task 2) Model 1 — User-defined baseline CNN\n\n"
        "- Conv2D(32, 3x3) -> ReLU -> MaxPool\n"
        "- Conv2D(64, 3x3) -> ReLU -> MaxPool\n"
        "- Conv2D(64, 3x3) -> ReLU -> Flatten -> Dense(128) -> Dropout(0.3) -> Softmax(10)\n"
        "- Optimizer: Adam, batch size 128, epochs <= 20, early stopping on validation accuracy.\n\n"
        "## Task 3) Model 2 — Stronger CNN\n\n"
        "- Conv2D(32) -> BN -> ReLU -> Conv2D(32) -> BN -> MaxPool\n"
        "- Conv2D(64) -> BN -> ReLU -> Conv2D(64) -> BN -> MaxPool\n"
        "- Conv2D(128) -> BN -> ReLU -> Conv2D(128) -> BN -> MaxPool\n"
        "- Flatten -> Dense(256) -> Dropout(0.4) -> Softmax(10)\n"
        "- Optimizer: Adam, batch size 128, epochs <= 20, early stopping.\n\n"
        "## Task 4) Validation & test evaluation\n\n"
        f"{table}\n\n"
        "### Confusion matrix interpretation\n\n"
        f"{confusion_text}\n\n"
        "### Dataset source and license\n\n"
        "- Source: Fashion-MNIST by Zalando Research.\n"
        "- Link: https://github.com/zalandoresearch/fashion-mnist\n"
        "- License: MIT License.\n\n"
        "### Appendix\n\n"
        "- Split declaration: fixed train/validation/test split used for all experiments.\n"
        "- Input resizing: all images resized from 28x28 to 32x32 before CNN input.\n"
        "- Augmentation: none added in this implementation to keep the comparison fair.\n"
        ,
        encoding="utf-8",
    )


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    X_train, y_train, X_test, y_test = load_dataset()
    X_train_resized = resize_images(X_train)
    X_test_resized = resize_images(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_resized,
        y_train,
        test_size=6000,
        stratify=y_train,
        random_state=SEED,
    )

    class_distribution_plot(y_tr, y_val, y_test)
    save_task1_summary(y_tr, y_val, y_test)

    baseline = baseline_model()
    stronger = stronger_model()

    results = {}
    results["baseline"] = evaluate_model(baseline, "User-Defined Baseline", X_tr, y_tr, X_val, y_val, X_test_resized, y_test)
    results["stronger"] = evaluate_model(stronger, "Stronger CNN", X_tr, y_tr, X_val, y_val, X_test_resized, y_test)

    write_report(results)

    print("Train/val/test sizes:")
    print(len(X_tr), len(X_val), len(X_test_resized))
    print("Baseline => best val", results["baseline"]["best_val_acc"], "epoch", results["baseline"]["best_epoch"], "test acc", results["baseline"]["test_acc"])
    print("Stronger => best val", results["stronger"]["best_val_acc"], "epoch", results["stronger"]["best_epoch"], "test acc", results["stronger"]["test_acc"])
    print(f"Artifacts saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
