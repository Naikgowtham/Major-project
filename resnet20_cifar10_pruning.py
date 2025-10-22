"""
ResNet-20 Magnitude-Based Pruning on CIFAR-10
This script demonstrates magnitude-based pruning on ResNet-20 trained on CIFAR-10 dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

from pruning.pruning_methods import MagnitudePruning, IterativeMagnitudePruning

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_cifar10():
    """Load CIFAR-10 dataset with standard augmentation"""
    print("Loading CIFAR-10 dataset...")
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return trainloader, testloader


class BasicBlock(nn.Module):
    """Basic Block for ResNet"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Each stage has 3 blocks for ResNet-20 (n=3, total layers = 6n+2 = 20)
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        
        self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def train_model(model, trainloader, epochs=20, lr=0.1):
    """Train the ResNet-20 model"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    # Learning rate schedule: decay at 50% and 75% of total epochs
    milestones = [int(epochs * 0.5), int(epochs * 0.75)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    print(f"Training for {epochs} epochs with lr={lr}, milestones={milestones}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if i % 100 == 99:
                print(f'  [Epoch {epoch+1}/{epochs}, Batch {i+1}] Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        # Print epoch summary
        train_acc = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} completed - Train Accuracy: {train_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.4f}')
        scheduler.step()
    
    print('Finished Training')
    return model


def evaluate_model(model, testloader):
    """Evaluate the model on test set"""
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    return accuracy


def count_parameters(model):
    """Count total and non-zero parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)
    return total_params, nonzero_params


def save_model(model, filename):
    """Save model state dictionary"""
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/{filename}')
    print(f"Model saved to models/{filename}")


def plot_pruning_results(results):
    """Plot accuracy vs sparsity"""
    os.makedirs('plots', exist_ok=True)
    
    ratios = [r['ratio'] for r in results]
    sparsities = [r['sparsity'] * 100 for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy vs pruning ratio
    ax1.plot(ratios, accuracies, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Pruning Ratio', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Pruning Ratio', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(accuracies) - 2, max(accuracies) + 2])
    
    # Plot accuracy vs sparsity
    ax2.plot(sparsities, accuracies, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Sparsity (%)', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Sparsity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([min(accuracies) - 2, max(accuracies) + 2])
    
    plt.tight_layout()
    plt.savefig('plots/pruning_results.png', dpi=300)
    print("Pruning results plot saved to plots/pruning_results.png")
    plt.close()


def main():
    # Load CIFAR-10 dataset
    trainloader, testloader = load_cifar10()
    
    # Create ResNet-20 model
    print("\nCreating ResNet-20 model...")
    model = ResNet20(num_classes=10)
    model.to(device)
    
    total_params, _ = count_parameters(model)
    print(f"Model created with {total_params:,} parameters")
    
    # Train the original model
    print("\n" + "="*80)
    print("TRAINING ORIGINAL RESNET-20 MODEL")
    print("="*80)
    trained_model = train_model(model, trainloader, epochs=30, lr=0.1)
    
    # Evaluate original model
    print("\n" + "="*80)
    print("EVALUATING ORIGINAL MODEL")
    print("="*80)
    original_accuracy = evaluate_model(trained_model, testloader)
    
    # Save original model
    save_model(trained_model, "resnet20_cifar10_original.pth")
    
    # Count parameters
    total_params, nonzero_params = count_parameters(trained_model)
    print(f"\nOriginal Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Non-zero parameters: {nonzero_params:,}")
    print(f"  Sparsity: {100 * (1 - nonzero_params/total_params):.2f}%")
    
    # Apply magnitude-based pruning with different ratios
    print("\n" + "="*80)
    print("MAGNITUDE-BASED PRUNING")
    print("="*80)
    
    pruning_ratios = [0.3, 0.5, 0.7, 0.9]
    results = []
    
    for ratio in pruning_ratios:
        print(f"\n{'-'*80}")
        print(f"Pruning with ratio: {ratio} ({ratio*100:.0f}%)")
        print(f"{'-'*80}")
        
        # Create a copy of the trained model
        pruned_model = ResNet20(num_classes=10)
        pruned_model.load_state_dict(trained_model.state_dict())
        pruned_model.to(device)
        
        # Apply magnitude pruning
        pruner = MagnitudePruning(pruned_model, prune_ratio=ratio)
        pruned_model = pruner.prune()
        
        # Get sparsity
        total_params, nonzero_params = count_parameters(pruned_model)
        sparsity = 1 - (nonzero_params / total_params)
        
        print(f"\nPruned Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-zero parameters: {nonzero_params:,}")
        print(f"  Sparsity: {sparsity*100:.2f}%")
        
        # Evaluate pruned model (before fine-tuning)
        print("\nEvaluating pruned model (before fine-tuning)...")
        pruned_accuracy_before = evaluate_model(pruned_model, testloader)
        accuracy_drop_before = original_accuracy - pruned_accuracy_before
        print(f"Accuracy drop: {accuracy_drop_before:.2f}%")
        
        # Fine-tune the pruned model
        print("\nFine-tuning the pruned model...")
        fine_tuned_model = train_model(pruned_model, trainloader, epochs=10, lr=0.01)
        
        # Evaluate after fine-tuning
        print("\nEvaluating pruned model (after fine-tuning)...")
        pruned_accuracy_after = evaluate_model(fine_tuned_model, testloader)
        accuracy_drop_after = original_accuracy - pruned_accuracy_after
        print(f"Accuracy drop after fine-tuning: {accuracy_drop_after:.2f}%")
        
        # Save pruned model
        save_model(fine_tuned_model, f"resnet20_cifar10_pruned_{int(ratio*100)}.pth")
        
        # Store results
        results.append({
            'ratio': ratio,
            'sparsity': sparsity,
            'accuracy_before': pruned_accuracy_before,
            'accuracy_after': pruned_accuracy_after,
            'accuracy': pruned_accuracy_after
        })
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"\nOriginal ResNet-20 Model:")
    print(f"  Accuracy: {original_accuracy:.2f}%")
    print(f"  Parameters: {total_params:,}")
    
    print(f"\n{'Prune Ratio':<15} {'Sparsity':<15} {'Acc (Before)':<15} {'Acc (After)':<15} {'Drop':<10}")
    print("-" * 70)
    for result in results:
        ratio = result['ratio']
        sparsity = result['sparsity'] * 100
        acc_before = result['accuracy_before']
        acc_after = result['accuracy_after']
        drop = original_accuracy - acc_after
        print(f"{ratio*100:>6.0f}%{'':<9} {sparsity:>6.2f}%{'':<9} {acc_before:>6.2f}%{'':<9} {acc_after:>6.2f}%{'':<9} {drop:>5.2f}%")
    
    # Plot results
    plot_pruning_results(results)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
