import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from get_dataset import GiMeFiveDataset
from models import GiMeFive, GiMeFiveRes, VGG, EmotionClassifierResNet18, EmotionClassifierResNet34

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved preprocessed model")
    parser.add_argument("--model", type=str, required=True,
                        choices=["gimefive", "gimefiveres", "vgg", "resnet18", "resnet34"])
    parser.add_argument("--dataset", type=str, default="fer2013",
                        choices=["fer2013"])
    return parser.parse_args()


def build_model(model_name: str):
    if model_name == "gimefive":
        return GiMeFive().to(device)
    elif model_name == "gimefiveres":
        return GiMeFiveRes().to(device)
    elif model_name == "vgg":
        return VGG().to(device)
    elif model_name == "resnet18":
        return EmotionClassifierResNet18().to(device)
    elif model_name == "resnet34":
        return EmotionClassifierResNet34().to(device)
    raise ValueError(f"Unknown model: {model_name}")


def main():
    args = parse_args()
    os.makedirs("results_preproc", exist_ok=True)

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = GiMeFiveDataset(
        csv_file="archive/FER2013/test_FER_labels.csv",
        img_dir="archive/FER2013/test_64x64/",
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = build_model(args.model)

    model_path = f"models_preproc/best_model_{args.model}_{args.dataset}_preproc.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_loss = running_loss / len(test_loader)
    test_acc = accuracy_score(all_labels, all_preds)

    class_names = ["happiness", "surprise", "sadness", "anger", "disgust", "fear"]

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    metrics_path = f"results_preproc/evaluation_metrics_{args.model}_{args.dataset}_preproc.txt"
    report_path = f"results_preproc/classification_report_{args.model}_{args.dataset}_preproc.txt"
    cm_path = f"results_preproc/confusion_matrix_{args.model}_{args.dataset}_preproc.png"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Accuracy: {test_acc:.6f}\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {args.model} on {args.dataset} preprocessed")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print("Evaluation terminée.")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_acc:.6f}")
    print("Fichiers générés :")
    print(f"- {metrics_path}")
    print(f"- {report_path}")
    print(f"- {cm_path}")


if __name__ == "__main__":
    main()