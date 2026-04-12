import torch
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import os
import argparse
import time
# import numpy as np
from matplotlib import pyplot as plt
# from sklearn.model_selection import ParameterGrid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

from get_dataset import GiMeFiveDataset
from models import GiMeFive, GiMeFiveRes
from models import SEBlock, ResidualBlock, BasicBlock
from models import VGG
from models import ResNet, EmotionClassifierResNet18, EmotionClassifierResNet34

def parse_args():
    parser = argparse.ArgumentParser(description="Train facial emotion recognition models")

    parser.add_argument("--model", type=str, default="gimefive",
                        choices=["gimefive", "gimefiveres", "vgg", "resnet18", "resnet34"],
                        help="Model architecture to train")

    parser.add_argument("--dataset", type=str, default="fer2013",
                        choices=["fer2013", "rafdb", "gimefive"],
                        help="Dataset configuration to use")

    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")

    return parser.parse_args()

def main():
    print(f"Using device: {device}")
    args = parse_args()
    start_time = time.time()
    os.makedirs("models_preproc", exist_ok=True)
    os.makedirs("reports_preproc", exist_ok=True)
    os.makedirs("results_preproc", exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(scale=(0.02,0.25)),
    ])
        
    if args.dataset == "fer2013":
        train_csv = 'archive/FER2013/train_FER_labels.csv'
        train_dir = 'archive/FER2013/train_64x64/'
        test_csv = 'archive/FER2013/test_FER_labels.csv'
        test_dir = 'archive/FER2013/test_64x64/'
        valid_csv = 'data/valid_labels.csv'
        valid_dir = 'data/valid_64x64/'

    elif args.dataset == "rafdb":
        train_csv = 'archive/RAF-DB/train_RAF_labels.csv'
        train_dir = 'archive/RAF-DB/train/'
        test_csv = 'archive/RAF-DB/test_RAF_labels.csv'
        test_dir = 'archive/RAF-DB/test/'
        valid_csv = 'data/valid_labels.csv'
        valid_dir = 'data/valid'

    elif args.dataset == "gimefive":
        train_csv = 'data/train_labels.csv'
        train_dir = 'data/train/'
        test_csv = 'data/test_labels.csv'
        test_dir = 'data/test/'
        valid_csv = 'data/valid_labels.csv'
        valid_dir = 'data/valid'

    # rafdb_dataset_train = GiMeFiveDataset(csv_file='archive/RAF-DB/train_RAF_labels.csv',
    #                             img_dir='archive/RAF-DB/train/',
    #                             transform=transform)

    rafdb_dataset_train = GiMeFiveDataset(
        csv_file=train_csv,
        img_dir=train_dir,
        transform=transform
    )

    # rafdb_dataset_train = GiMeFiveDataset(csv_file='data/train_labels.csv',
    #                             img_dir='data/train/',
    #                             transform=transform)
    data_train_loader = DataLoader(rafdb_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_image, train_label = next(iter(data_train_loader))
    print(f"Train batch: image shape {train_image.shape}, labels shape {train_label.shape}")

    rafdb_dataset_vali = GiMeFiveDataset(
        csv_file=valid_csv,
        img_dir=valid_dir,
        transform=transform
    )
    data_vali_loader = DataLoader(rafdb_dataset_vali, batch_size=args.batch_size, shuffle=False, num_workers=0)
    vali_image, vali_label = next(iter(data_vali_loader))
    print(f"Vali batch: image shape {vali_image.shape}, labels shape {vali_label.shape}")

    # rafdb_dataset_test = GiMeFiveDataset(csv_file='archive/RAF-DB/test_RAF_labels.csv',
    #                             img_dir='archive/RAF-DB/test/',
    #                             transform=transform)

    rafdb_dataset_test = GiMeFiveDataset(
        csv_file=test_csv,
        img_dir=test_dir,
        transform=transform
    )

    # rafdb_dataset_test = GiMeFiveDataset(csv_file='data/test_labels.csv',
    #                             img_dir='data/test/',
    #                             transform=transform)
    data_test_loader = DataLoader(rafdb_dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_image, test_label = next(iter(data_test_loader))
    print(f"Test batch: image shape {test_image.shape}, labels shape {test_label.shape}")


    if args.model == "gimefive":
        model = GiMeFive().to(device)
    elif args.model == "gimefiveres":
        model = GiMeFiveRes().to(device)
    elif args.model == "vgg":
        model = VGG().to(device)
    elif args.model == "resnet18":
        model = EmotionClassifierResNet18().to(device)
    elif args.model == "resnet34":
        model = EmotionClassifierResNet34().to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    patience = 15
    best_val_acc = 0  
    patience_counter = 0

    num_epochs = args.epochs

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []
    

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in tqdm(data_train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(data_train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in data_test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss = test_running_loss / len(data_test_loader)
        test_acc = test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data in data_vali_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(data_vali_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {test_loss}, Test Accuracy: {test_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 
            torch.save(model.state_dict(), f'models_preproc/best_model_{args.model}_{args.dataset}_preproc.pth')
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epochs.")
        
        if patience_counter > patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break

    # Plotting and saving results

        # Plotting and saving results
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves on GiMeFive')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'reports_preproc/loss_curve_{args.model}_{args.dataset}_preproc.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves on GiMeFive')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'reports_preproc/accuracy_curve_{args.model}_{args.dataset}_preproc.png', dpi=300)
    plt.close()

    df = pd.DataFrame({
        'Epoch': list(epochs),
        'Train Loss': train_losses,
        'Test Loss': test_losses,
        'Validation Loss': val_losses,
        'Train Accuracy': train_accuracies,
        'Test Accuracy': test_accuracies,
        'Validation Accuracy': val_accuracies
    })
    df.to_csv(f'reports_preproc/training_metrics_{args.model}_{args.dataset}_preproc.csv', index=False)

    training_time = time.time() - start_time
    best_val_accuracy = max(val_accuracies) if val_accuracies else None
    epochs_executed = len(train_losses)

    summary_df = pd.DataFrame([{
        "model": args.model,
        "dataset": args.dataset,
        "epochs_requested": args.epochs,
        "epochs_executed": epochs_executed,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_parameters": total_params,
        "best_val_accuracy": best_val_accuracy,
        "final_train_accuracy": train_accuracies[-1] if train_accuracies else None,
        "final_test_accuracy": test_accuracies[-1] if test_accuracies else None,
        "final_validation_accuracy": val_accuracies[-1] if val_accuracies else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_test_loss": test_losses[-1] if test_losses else None,
        "final_validation_loss": val_losses[-1] if val_losses else None,
        "training_time_seconds": training_time
    }])

    summary_df.to_csv(
        f"reports_preproc/run_summary_{args.model}_{args.dataset}.csv",
        index=False
    )

    print("Saved files:")
    print(f"- reports_preproc/run_summary_{args.model}_{args.dataset}.csv")
    print(f"- models_preproc/best_model_{args.model}_{args.dataset}.pth")
    print(f"- reports_preproc/training_metrics_{args.model}_{args.dataset}.csv")
    print(f"- reports_preproc/loss_curve_{args.model}_{args.dataset}.png")
    print(f"- reports_preproc/accuracy_curve_{args.model}_{args.dataset}.png")


if __name__ == '__main__':
    main()