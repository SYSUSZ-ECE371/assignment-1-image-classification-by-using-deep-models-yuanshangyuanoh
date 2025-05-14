import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR,LambdaLR
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import os
import time
import copy

#冻结模型中的某些层
def freeze_layers(model, layer_names):
    for name, param in model.named_parameters():
        if name in layer_names:
            param.requires_grad = False  # 冻结该层
        else:
            param.requires_grad = True   # 其他层正常训练

if __name__ == '__main__':
    # Set data directory
    data_dir = 'data/flower_dataset'

    # Data augmentation and normalization for training and validation
    data_transforms = transforms.Compose([
        # GRADED FUNCTION: Add five data augmentation methods, Normalizating and Tranform to tensor
        ### START SOLUTION HERE ###
        transforms.RandomResizedCrop(224),  # 随机裁剪并调整大小
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.RandomRotation(30),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 添加颜色扰动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(p=0.2),  # 随机擦除
        # Add five data augmentation methods, Normalizating and Tranform to tensor
        ### END SOLUTION HERE ###
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(data_dir, data_transforms)

    # Automatically split into 80% train and 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Use DataLoader for both train and validation datasets
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Get class names from the dataset
    class_names = full_dataset.classes

    # Load pre-trained model and modify the last layer
    model = models.resnet101(pretrained=True)
    freeze_layers(model, ['layer1','layer2'])

    # GRADED FUNCTION: Modify the last fully connected layer of model
    ### START SOLUTION HERE ###
    # Modify the last fully connected layer of model
    num_in_features = model.fc.in_features
    # model.fc = nn.Linear(num_in_features, len(class_names))
    model.fc = nn.Sequential(
        nn.Linear(num_in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),#防止过拟合
        nn.Linear(256, len(class_names))
    )
    ### END SOLUTION HERE ###


    # GRADED FUNCTION: Define the loss function
    ### START SOLUTION HERE ###
    # Define the loss function
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    ### END SOLUTION HERE ###

    # GRADED FUNCTION: Define the optimizer
    ### START SOLUTION HERE ###
    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    ### END SOLUTION HERE ###

    # Learning rate scheduler
    warmup_epochs = 5
    num_epochs = 100
    warmup_lr_init = 0.001
    final_lr = 0.01
    # scheduler = LinearLR(
    #     optimizer,
    #     start_factor = 1,
    #     end_factor = 0.1,
    # )

    linearLR = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_lr_init/final_lr + (1 - warmup_lr_init/final_lr) * (epoch / warmup_epochs)
    )
    cosine_annealing = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[linearLR, cosine_annealing],
        milestones=[warmup_epochs]
    )

    # Training function
    def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Print learning rate for current epoch
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning Rate: {current_lr:.6f}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':

                            # GRADED FUNCTION: Backward pass and optimization
                            ### START SOLUTION HERE ###
                            # Backward pass and optimization
                            loss.backward()  # 反向传播
                            # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                            optimizer.step()  # 参数更新
                            ### END SOLUTION HERE ###

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()  # Update learning rate based on scheduler

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Save the model if validation accuracy is the best so far
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save the best model
                    save_dir = 'Ex2/work_dir'
                    os.makedirs(save_dir, exist_ok=True)

                    # GRADED FUNCTION: Save the best model
                    ### START SOLUTION HERE ###
                    # Save the best model
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))  # 保存最佳模型权重
                    ### END SOLUTION HERE ###

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(best_model_wts)
        return model


    # Train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=50)

    # C:\Users\86155/.cache\torch\hub\checkpoints\resnet101-63fe2227.pth
    # C:\Users\86155/.cache\torch\hub\checkpoints\resnet152-394f9c45.pth
    # C:\Users\86155 /.cache\torch\hub\checkpoints\resnet18 - f37072fd.pth