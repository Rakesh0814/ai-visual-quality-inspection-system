from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
VAL_DIR = ROOT / "dataset" / "val"

MODEL_PATH = ROOT / "app" / "models" / "pytorch_quality_model.pt"
META_PATH = ROOT / "app" / "models" / "pytorch_quality_model.json"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 12
LEARNING_RATE = 1e-4


class QualityClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.DEFAULT

        self.backbone = models.mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[-1].in_features

        self.backbone.classifier[-1] = nn.Linear(
            in_features,
            1
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def main():

    print("\nChecking dataset...")

    required = [
        TRAIN_DIR / "good",
        TRAIN_DIR / "defect",
        VAL_DIR / "good",
        VAL_DIR / "defect",
    ]

    for folder in required:
        if not folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {folder}"
            )

    print("Dataset folders found.")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(8),

        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.05
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_transform
    )

    print(
        "Class mapping:",
        train_dataset.class_to_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = QualityClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    defect_index = train_dataset.class_to_idx[
        "defect"
    ]

    best_accuracy = 0.0

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)

            binary_labels = (
                labels == defect_index
            ).float()

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                binary_labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss /
            max(1, len(train_loader))
        )

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                labels = labels.to(device)

                binary_labels = (
                    labels == defect_index
                ).float()

                logits = model(images)

                probabilities = torch.sigmoid(
                    logits
                )

                predictions = (
                    probabilities >= 0.5
                ).float()

                correct += (
                    predictions ==
                    binary_labels
                ).sum().item()

                total += labels.size(0)

        validation_accuracy = (
            correct / total
            if total > 0
            else 0
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation Accuracy: "
            f"{validation_accuracy:.2%}"
        )

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            MODEL_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "  -> Best model saved"
            )

    metadata = {
        "architecture": "MobileNetV3-Small",
        "framework": "PyTorch",
        "dataset": "Custom plastic bottle dataset",
        "task": "good vs defect bottle classification",
        "validation_accuracy": round(
            best_accuracy,
            4
        ),
        "image_size": IMAGE_SIZE,
        "threshold": 0.5
    }

    META_PATH.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nTraining complete.")
    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.2%}"
    )
    print(
        f"Model saved to:\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()