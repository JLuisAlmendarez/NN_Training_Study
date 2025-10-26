import os
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, models
from torchvision.datasets import MNIST, ImageFolder
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output

def download_mnist_data(root='./data'):
    transform = transforms.ToTensor()
    train_data = MNIST(root=root, train=True, download=True, transform=transform)
    test_data = MNIST(root=root, train=False, download=True, transform=transform)
    print(f"Train dataset size: {len(train_data)}")
    print(f"Test dataset size: {len(test_data)}")
    return train_data, test_data

def create_directory_structure(base_dir='./data/MNIST/use/original'):
    for split in ['train', 'test']:
        path = os.path.join(base_dir, split)
        os.makedirs(path, exist_ok=True)
    print(f"Directory structure created at {base_dir}")

def save_subset(dataset, split_dir, n_per_class=10, train_ratio=0.8):
    counts = {i: 0 for i in range(10)}
    for img, label in dataset:
        if counts[label] < n_per_class:
            counts[label] += 1
            target_split = 'train' if counts[label] <= n_per_class * train_ratio else 'test'
            save_dir = os.path.join(split_dir, target_split, str(label))
            os.makedirs(save_dir, exist_ok=True)
            img_path = os.path.join(save_dir, f'{label}_{counts[label]}.jpg')
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(img_path)
    print(f"Saved {n_per_class} images per class to {split_dir}")

def count_images(base_dir, split):
    split_path = os.path.join(base_dir, split)
    num_images = sum(
        len([f for f in os.listdir(os.path.join(split_path, cls)) if f.endswith('.jpg')])
        for cls in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, cls))
    )
    return num_images

def get_augmentation_transform():
    augment_transform = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])
    return augment_transform

def generate_augmented(input_dir, output_dir, target_count):
    augment_transform = get_augmentation_transform()
    os.makedirs(output_dir, exist_ok=True)
    classes = [cls for cls in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, cls))]
    for cls in classes:
        input_class_dir = os.path.join(input_dir, cls)
        output_class_dir = os.path.join(output_dir, cls)
        os.makedirs(output_class_dir, exist_ok=True)
        images = [f for f in os.listdir(input_class_dir) if f.endswith('.jpg')]
        current_count = 0
        counter = 0
        while current_count < target_count:
            img_name = random.choice(images)
            img = Image.open(os.path.join(input_class_dir, img_name))
            aug_img = augment_transform(img)
            counter += 1
            new_name = f"{img_name.split('.')[0]}_aug{counter}.jpg"
            aug_img.save(os.path.join(output_class_dir, new_name))
            current_count += 1
    print(f"Generated {target_count} augmented images per class in {output_dir}")

def plot_images(base_dir, titulo):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()
    clases = sorted([cls for cls in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, cls))], key=int)
    for idx, cls in enumerate(clases):
        class_dir = os.path.join(base_dir, cls)
        img_name = sorted([f for f in os.listdir(class_dir) if f.endswith('.jpg')])[0]
        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img)
        axes[idx].imshow(img_tensor.squeeze(), cmap='gray')
        axes[idx].set_title(f"Clase {cls}")
        axes[idx].axis('off')

    fig.suptitle(titulo, fontsize=16)
    plt.tight_layout()
    plt.show()

def get_train_transform():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transform

def get_test_transform():
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return test_transform

def create_combined_dataset(original_path, augmented_path, transform, split='train'):
    dataset = ConcatDataset([
        ImageFolder(original_path, transform=transform),
        ImageFolder(augmented_path, transform=transform)
    ])
    print(f"{split.capitalize()} dataset size: {len(dataset)}")
    return dataset

def create_train_test_datasets(base_original='./data/MNIST/use/original',
                               base_augmented='./data/MNIST/use/augmented'):
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    train_dataset = create_combined_dataset(
        os.path.join(base_original, 'train'),
        os.path.join(base_augmented, 'train'),
        train_transform,
        split='train'
    )
    test_dataset = create_combined_dataset(
        os.path.join(base_original, 'test'),
        os.path.join(base_augmented, 'test'),
        test_transform,
        split='test'
    )
    print(f"Total images: {len(train_dataset) + len(test_dataset)}")
    return train_dataset, test_dataset

def get_model(model_name="alexnet", num_classes=10, weight_init="kaiming", dropout_rate=0.0, input_size=(1, 224, 224)):
    if model_name.lower() == "alexnet":
        model = models.alexnet(weights=None)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        in_features = model.classifier[-1].in_features
        layers = list(model.classifier.children())[:-1]
        if dropout_rate > 0:
            layers.insert(-1, nn.Dropout(dropout_rate))
        layers.append(nn.Linear(in_features, num_classes))
        model.classifier = nn.Sequential(*layers)
    elif model_name.lower() == "vgg":
        model = models.vgg11(weights=None)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        in_features = model.classifier[-1].in_features
        layers = list(model.classifier.children())[:-1]
        if dropout_rate > 0:
            layers.insert(-1, nn.Dropout(dropout_rate))
        layers.append(nn.Linear(in_features, num_classes))
        model.classifier = nn.Sequential(*layers)
    elif model_name.lower() == "customcnn":
        class CustomCNN(nn.Module):
            def __init__(self, num_classes=10, dropout_rate=0.0, input_size=(1, 224, 224)):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
                )
                with torch.no_grad():
                    x = torch.zeros(1, *input_size)
                    x = self.features(x)
                    flatten_size = x.view(1, -1).shape[1]
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flatten_size, 256), nn.ReLU(),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    nn.Linear(256, 128), nn.ReLU(),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    nn.Linear(128, num_classes)
                )
            def forward(self, x):
                return self.classifier(self.features(x))
        model = CustomCNN(num_classes=num_classes, dropout_rate=dropout_rate, input_size=input_size)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if weight_init.lower() == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif weight_init.lower() == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif weight_init.lower() == "orthogonal":
                nn.init.orthogonal_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    print(f"Model '{model_name}' created with {weight_init} initialization")
    return model

def set_constant_with_warmup(optimizer, base_lr, epoch, warmup_epochs=5):
    if epoch <= warmup_epochs and warmup_epochs > 0:
        lr = base_lr * epoch / float(warmup_epochs)
    else:
        lr = base_lr
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr

def lr_range_test(model, train_loader, optimizer_class, lr_start=1e-6, lr_end=1, num_iters=100, device="cpu"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), lr=lr_start)
    lrs, losses = [], []
    mult = (lr_end / lr_start) ** (1 / num_iters)
    lr = lr_start
    iterator = iter(train_loader)
    for i in range(num_iters):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels = next(iterator)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lrs.append(lr)
        losses.append(loss.item())
        lr *= mult
        for g in optimizer.param_groups:
            g['lr'] = lr
    plt.figure(figsize=(6, 4))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("LR Range Test")
    plt.show()
    return lrs, losses

def train_model(model_name,
                train_dataset, test_dataset,
                batch_size=128,
                optimizer_name="SGD",
                nesterov=False,
                lr=0.01,
                weight_decay=5e-4,
                epochs=60,
                dropout_rate=0.0,
                weight_init="kaiming",
                label_smoothing=0.0,
                lr_schedule="constant+warmup",
                use_augment=False,
                protocol="Fixed Epochs",
                wall_clock_budget=None,
                seed=42,
                device=None,
                num_workers=2,
                pin_memory=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Usando dispositivo: CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Usando dispositivo: MPS")
        else:
            device = torch.device("cpu")
            print("Usando dispositivo: CPU")
    if use_augment:
        extra = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
        ])
        if hasattr(train_dataset, "transform") and train_dataset.transform is not None:
            train_dataset.transform = transforms.Compose([extra, train_dataset.transform])
        else:
            train_dataset.transform = extra
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    model = get_model(model_name=model_name, num_classes=10,
                      weight_init=weight_init, dropout_rate=dropout_rate)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if optimizer_name.lower() == "sgd" or optimizer_name.lower() == "sgd+nesterov":
        use_nesterov = (optimizer_name.lower() == "sgd+nesterov") or nesterov
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay, nesterov=use_nesterov)
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizador desconocido: {optimizer_name}")
    if lr_schedule.lower() == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    elif lr_schedule.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_schedule.lower() == "one_cycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                  steps_per_epoch=len(train_loader), epochs=epochs)
    else:
        scheduler = None
    logs = []
    best_val_acc = 0.0
    best_model_state = None
    epoch_times = []
    epoch_lrs = []
    start_time_global = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        if lr_schedule.lower() == "constant+warmup":
            current_lr = set_constant_with_warmup(optimizer, lr, epoch, warmup_epochs=5)
        else:
            current_lr = optimizer.param_groups[0]['lr']
        epoch_lrs.append(current_lr)
        model.train()
        train_loss_accum, train_correct, train_total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if lr_schedule.lower() == "one_cycle":
                scheduler.step()
            train_loss_accum += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)
        train_loss = train_loss_accum / train_total
        train_acc = train_correct / train_total
        model.eval()
        test_loss_accum, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss_accum += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                test_correct += preds.eq(labels).sum().item()
                test_total += labels.size(0)
        test_loss = test_loss_accum / test_total
        test_acc = test_correct / test_total
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            best_model_state = model.state_dict().copy()
        if scheduler and lr_schedule.lower() not in ["constant+warmup", "one_cycle"]:
            scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": current_lr,
            "epoch_time_s": epoch_time
        })
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
        if protocol == "Fixed Wall-Clock Time" and wall_clock_budget:
            if time.time() - start_time_global >= wall_clock_budget:
                print(f"\n⏰ Wall-clock budget alcanzado ({wall_clock_budget}s). Deteniendo...")
                break
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.eval()
        test_loss_accum_final, test_correct_final, test_total_final = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss_accum_final += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                test_correct_final += preds.eq(labels).sum().item()
                test_total_final += labels.size(0)
    final_test_loss = test_loss_accum_final / test_total_final
    final_test_acc = test_correct_final / test_total_final
    print(f"\n🧪 Test final — Loss: {final_test_loss:.4f} | Acc: {final_test_acc:.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([l["epoch"] for l in logs], [l["train_loss"] for l in logs], label="train_loss")
    axes[0].plot([l["epoch"] for l in logs], [l["test_loss"] for l in logs], label="test_loss")
    axes[0].set_title("Loss por época")
    axes[0].legend()
    axes[1].plot([l["epoch"] for l in logs], [l["train_acc"] for l in logs], label="train_acc")
    axes[1].plot([l["epoch"] for l in logs], [l["test_acc"] for l in logs], label="test_acc")
    axes[1].set_title("Accuracy por época")
    axes[1].legend()
    plt.show()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    model_dir = results_dir / model_name.lower()
    model_dir.mkdir(exist_ok=True)
    exp_idx = len(list(model_dir.glob("exp_*"))) + 1
    exp_path = model_dir / f"exp_{exp_idx}"
    exp_path.mkdir(exist_ok=True)
    metrics_df = pd.DataFrame(logs)
    metrics_df.to_json(exp_path / "metrics.json", index=False)
    metadata = {
        "model_name": model_name,
        "optimizer": optimizer_name,
        "nesterov_flag": True if optimizer_name.lower() == "sgd+nesterov" else False,
        "lr": lr,
        "weight_decay": weight_decay,
        "weight_init": weight_init,
        "dropout_rate": dropout_rate,
        "label_smoothing": label_smoothing,
        "lr_schedule": lr_schedule,
        "batch_size": batch_size,
        "use_augment": use_augment,
        "protocol": protocol,
        "wall_clock_budget_s": wall_clock_budget,
        "seed": seed,
        "device": str(device),
        "num_epochs_run": len(logs),
        "total_train_time_s": sum(epoch_times),
        "epoch_times_s": epoch_times,
        "epoch_lrs": epoch_lrs,
        "best_val_acc": float(best_val_acc),
        "final_test_acc": float(final_test_acc),
        "final_test_loss": float(final_test_loss)
    }
    with open(exp_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    torch.save(best_model_state, exp_path / "model.pth")
    metadata["model_file"] = str((exp_path / "model.pth").resolve())
    with open(exp_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n✅ Modelo y métricas guardados en: {exp_path}")
    print(f"✓ Mejor test accuracy: {best_val_acc:.4f}")
    return exp_path

def create_training_ui(train_dataset, test_dataset):
    model_widget = widgets.Dropdown(
        options=["alexnet", "vgg", "customcnn"],
        value="alexnet",
        description="Modelo"
    )
    optimizer_widget = widgets.Dropdown(
        options=["SGD", "SGD+Nesterov", "Adam", "AdamW"],
        value="SGD",
        description="Optimizer"
    )
    nesterov_checkbox = widgets.Checkbox(
        value=True,
        description="Nesterov (when applicable)"
    )
    nesterov_checkbox.layout.display = "none"

    def on_opt_change(change):
        if change["new"].lower() == "sgd+nesterov":
            nesterov_checkbox.layout.display = "block"
        else:
            nesterov_checkbox.layout.display = "none"
    optimizer_widget.observe(on_opt_change, names="value")
    on_opt_change({"new": optimizer_widget.value})
    batch_widget = widgets.Dropdown(
        options=[32, 128, 512],
        value=128,
        description="Batch size"
    )
    lr_widget = widgets.FloatLogSlider(
        value=0.01,
        base=10,
        min=-4,
        max=-1,
        step=0.1,
        description="LR"
    )
    weight_decay_widget = widgets.FloatSlider(
        value=5e-4,
        min=0.0,
        max=0.01,
        step=1e-4,
        description="Weight Decay"
    )
    dropout_widget = widgets.Checkbox(
        value=False,
        description="Dropout"
    )
    dropout_rate_widget = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=0.9,
        step=0.05,
        description="Dropout Rate"
    )
    weight_init_widget = widgets.Dropdown(
        options=["kaiming", "xavier", "orthogonal"],
        value="kaiming",
        description="Weight Init"
    )
    lr_schedule_widget = widgets.Dropdown(
        options=["constant+warmup", "step", "cosine", "one_cycle"],
        value="constant+warmup",
        description="LR Schedule"
    )
    label_smooth_widget = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=0.2,
        step=0.01,
        description="Label Smoothing"
    )
    augment_widget = widgets.Checkbox(
        value=False,
        description="Augment Extra"
    )
    protocol_widget = widgets.Dropdown(
        options=["Fixed Epochs", "Fixed Wall-Clock Time"],
        value="Fixed Epochs",
        description="Protocol"
    )
    epochs_box = widgets.BoundedIntText(
        value=60,
        min=1,
        max=100,
        description="Epochs"
    )
    time_box = widgets.BoundedIntText(
        value=90,
        min=1,
        max=120,
        description="Time (min)"
    )
    time_box.layout.display = "none"
    seed_box = widgets.BoundedIntText(
        value=42,
        min=0,
        max=9999,
        description="Seed"
    )

    def on_protocol_change(change):
        if change["new"] == "Fixed Epochs":
            epochs_box.layout.display = "block"
            time_box.layout.display = "none"
        else:
            epochs_box.layout.display = "none"
            time_box.layout.display = "block"
    protocol_widget.observe(on_protocol_change, names="value")
    on_protocol_change({"new": protocol_widget.value})
    train_button = widgets.Button(
        description="🚀 Entrenar",
        button_style="success"
    )
    lrtest_button = widgets.Button(
        description="🔎 LR Range Test",
        button_style="warning"
    )
    output_widget = widgets.Output(layout={'border': '1px solid black'})

    def on_lrtest_clicked(btn):
        with output_widget:
            clear_output(wait=True)
            print("🔎 Ejecutando LR Range Test...\n")
            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            model = get_model(
                model_name=model_widget.value,
                num_classes=10,
                weight_init=weight_init_widget.value,
                dropout_rate=dropout_rate_widget.value if dropout_widget.value else 0.0
            )
            model.to(device)
            optimizer_class = {
                "SGD": lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9),
                "SGD+Nesterov": lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9, nesterov=True),
                "Adam": lambda params, lr: optim.Adam(params, lr=lr),
                "AdamW": lambda params, lr: optim.AdamW(params, lr=lr)
            }[optimizer_widget.value]
            train_loader = DataLoader(train_dataset, batch_size=int(batch_widget.value), shuffle=True)
            lr_range_test(model, train_loader, optimizer_class, device=device)

    def on_train_clicked(btn):
        with output_widget:
            clear_output(wait=True)
            print("🚀 Iniciando entrenamiento...\n")
            wall_budget = None
            if protocol_widget.value == "Fixed Wall-Clock Time":
                wall_budget = time_box.value * 60
            try:
                exp_path = train_model(
                    model_name=model_widget.value,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    batch_size=int(batch_widget.value),
                    optimizer_name=optimizer_widget.value,
                    nesterov=bool(nesterov_checkbox.value),
                    lr=float(lr_widget.value),
                    weight_decay=float(weight_decay_widget.value),
                    epochs=int(epochs_box.value),
                    dropout_rate=float(dropout_rate_widget.value) if dropout_widget.value else 0.0,
                    weight_init=weight_init_widget.value,
                    label_smoothing=float(label_smooth_widget.value),
                    lr_schedule=lr_schedule_widget.value,
                    use_augment=bool(augment_widget.value),
                    protocol=protocol_widget.value,
                    wall_clock_budget=wall_budget,
                    seed=int(seed_box.value),
                    device=None,
                    num_workers=2,
                    pin_memory=True
                )
                print(f"\n🏁 Experimento completado. Archivos en: {exp_path}")
            except Exception as e:
                print(f"❌ Error durante entrenamiento: {e}")
                import traceback
                traceback.print_exc()
    train_button.on_click(on_train_clicked)
    lrtest_button.on_click(on_lrtest_clicked)
    ui = widgets.VBox([
        widgets.HBox([model_widget, optimizer_widget, nesterov_checkbox]),
        widgets.HBox([batch_widget, lr_widget, weight_decay_widget]),
        widgets.HBox([dropout_widget, dropout_rate_widget, weight_init_widget]),
        widgets.HBox([lr_schedule_widget, label_smooth_widget, augment_widget]),
        widgets.HBox([protocol_widget, epochs_box, time_box, seed_box]),
        widgets.HBox([train_button, lrtest_button]),
        output_widget
    ])
    return ui
