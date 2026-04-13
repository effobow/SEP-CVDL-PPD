import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from get_dataset import GiMeFiveDataset
from models import GiMeFive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    os.makedirs("results", exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    #test_dataset = GiMeFiveDataset(
    #    csv_file='archive/FER2013/test_FER_labels.csv',
    #    img_dir='archive/FER2013/test/',
    #    transform=transform
    #)

    #test_dataset = GiMeFiveDataset(
    #    csv_file='archive/EXPW/expw_labels.csv',
    #    img_dir='archive/EXPW/images/',
    #    transform=transform
    #)

    test_dataset = GiMeFiveDataset(
        csv_file='archive/JAFFE/jaffe_labels.csv',
        img_dir='archive/JAFFE/images/',
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = GiMeFive().to(device)
    #model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    #model.load_state_dict(torch.load('models/best_model_gimefive_fer2013.pth', map_location=device))
    model.load_state_dict(torch.load('models/best_model_gimefive_fer2013.pth', map_location=device))
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
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    with open("results/evaluation_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Accuracy: {test_acc:.6f}\n")

    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - GiMeFive")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=300)
    plt.close()

    print("Evaluation terminée.")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_acc:.6f}")
    print("Fichiers générés :")
    print("- evaluation_metrics.txt")
    print("- classification_report.txt")
    print("- confusion_matrix.png")

if __name__ == "__main__":
    main()