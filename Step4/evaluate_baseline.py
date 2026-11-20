import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "baseline_svm_model.joblib"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_features(image_path: Path, image_size=(128, 128)) -> np.ndarray:
    """Keep preprocessing identical to baseline_model.py."""
    image = Image.open(image_path).convert("RGB").resize(tuple(image_size))

    gray_image = np.array(image.convert("L"))
    hog_features = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    hsv_image = np.array(image.convert("HSV"))
    hsv_features = []
    for channel in range(3):
        histogram, _ = np.histogram(
            hsv_image[:, :, channel], bins=32, range=(0, 256)
        )
        histogram = histogram.astype("float32")
        histogram /= histogram.sum() + 1e-7
        hsv_features.extend(histogram)

    return np.concatenate([hog_features, np.asarray(hsv_features)])


def load_split(dataset_dir: Path, image_size):
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    class_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in: {dataset_dir}")

    features, labels, skipped = [], [], []
    counts = {}

    for class_dir in class_dirs:
        count = 0
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                features.append(extract_features(image_path, image_size))
                labels.append(class_dir.name)
                count += 1
            except Exception as exc:
                skipped.append({"file": str(image_path), "error": str(exc)})
        counts[class_dir.name] = count

    if not labels:
        raise ValueError(f"No readable images found in: {dataset_dir}")

    return np.asarray(features), np.asarray(labels), counts, skipped


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved SVM baseline.")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Dataset split to evaluate (default: validation).",
    )
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}\n"
            "Run baseline_model.py first."
        )

    dataset_dir = BASE_DIR / "data" / "processed" / args.split
    output_dir = BASE_DIR / "results" / "evaluation" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    class_names = list(bundle["classes"])
    image_size = bundle.get("image_size", (128, 128))

    print(f"\nLoading {args.split} dataset from: {dataset_dir}")
    X, y_true, counts, skipped = load_split(dataset_dir, image_size)

    unknown = sorted(set(y_true) - set(class_names))
    if unknown:
        raise ValueError(f"Dataset contains classes absent from training: {unknown}")

    y_pred = model.predict(X)
    accuracy = float(accuracy_score(y_true, y_pred))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    metrics = {
        "split": args.split,
        "dataset_path": str(dataset_dir),
        "number_of_images": int(len(y_true)),
        "number_of_features": int(X.shape[1]),
        "class_names": class_names,
        "images_per_class": counts,
        "accuracy": accuracy,
        "classification_report": report_dict,
        "skipped_images": skipped,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        [{"actual": actual, "predicted": predicted}
         for actual, predicted in zip(y_true, y_pred)]
    ).to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(report_dict).transpose().to_csv(
        output_dir / "classification_report.csv", encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"SVM Baseline - {args.split.title()} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print("\nEvaluation completed.")
    print(f"Split: {args.split}")
    print(f"Images: {len(y_true)}")
    print(f"Features per image: {X.shape[1]}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification report:")
    print(report_text)
    print(f"\nResults saved to: {output_dir}")
    if skipped:
        print(f"Warning: {len(skipped)} image(s) could not be read; see metrics.json.")


if __name__ == "__main__":
    main()
