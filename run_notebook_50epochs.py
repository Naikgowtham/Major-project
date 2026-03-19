import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResNet-56 Architecture (from notebook)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet56(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet56, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 9, stride=1)
        self.layer2 = self._make_layer(32, 9, stride=2)
        self.layer3 = self._make_layer(64, 9, stride=2)
        self.linear = nn.Linear(64, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# MAPPruner from notebook (sigmoid-based attention, linear schedule)
class MAPPruner:
    def __init__(self, model, target_sparsity=0.9, mask_update_freq=16, 
                 total_epochs=50, exploration_end=37, exploitation_start=45):
        self.model = model
        self.target_sparsity = target_sparsity
        self.mask_update_freq = mask_update_freq
        self.total_epochs = total_epochs
        self.exploration_end = exploration_end
        self.exploitation_start = exploitation_start
        
        self.masks = {}
        self.attention_values = {}
        self.current_sparsity = 0.0
        self.iteration_count = 0
        self.mask_frozen = False
        
        self._initialize_masks()
        self.sparsity_history = []
        self.mask_update_history = []
        
        print(f"[MAP Notebook] Initialized with sigmoid-based attention, linear schedule")
        print(f"  Target Sparsity: {target_sparsity * 100:.1f}%")
    
    def _initialize_masks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.masks[name] = torch.ones_like(module.weight.data)
                self.attention_values[name] = torch.abs(module.weight.data.clone())
    
    def _calculate_global_threshold(self, sparsity_ratio):
        all_magnitudes = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in self.masks:
                magnitudes = torch.abs(module.weight.data).view(-1)
                all_magnitudes.append(magnitudes)
        
        if not all_magnitudes:
            return 0.0
        all_magnitudes = torch.cat(all_magnitudes)
        k = int(len(all_magnitudes) * sparsity_ratio)
        if k >= len(all_magnitudes):
            return float('inf')
        if k <= 0:
            return 0.0
        threshold, _ = torch.kthvalue(all_magnitudes, k)
        return threshold.item()
    
    def _update_masks(self, current_epoch):
        if self.mask_frozen:
            return
        
        # LINEAR schedule (notebook implementation)
        if current_epoch < self.exploration_end:
            progress = current_epoch / self.exploration_end
            current_target_sparsity = self.target_sparsity * progress
        else:
            current_target_sparsity = self.target_sparsity
        
        threshold = self._calculate_global_threshold(current_target_sparsity)
        
        total_weights = 0
        pruned_weights = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in self.masks:
                # SIGMOID attention (notebook implementation)
                magnitude = torch.abs(module.weight.data)
                alpha = 1.0
                attention = torch.sigmoid(alpha * magnitude)
                self.attention_values[name] = attention
                
                self.masks[name] = (magnitude > threshold).float()
                
                total_weights += module.weight.numel()
                pruned_weights += (self.masks[name] == 0).sum().item()
        
        self.current_sparsity = pruned_weights / total_weights if total_weights > 0 else 0.0
        self.sparsity_history.append(self.current_sparsity)
        
        self.mask_update_history.append({
            'epoch': current_epoch,
            'iteration': self.iteration_count,
            'sparsity': self.current_sparsity,
            'threshold': threshold,
            'target_sparsity': current_target_sparsity
        })
    
    def apply_masks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in self.masks:
                module.weight.data = module.weight.data * self.masks[name]
    
    def step(self, current_epoch):
        self.iteration_count += 1
        if current_epoch >= self.exploitation_start:
            self.mask_frozen = True
        if (self.iteration_count % self.mask_update_freq == 0 and 
            not self.mask_frozen and 
            current_epoch < self.exploration_end):
            self._update_masks(current_epoch)
        self.apply_masks()
    
    def get_statistics(self):
        total_params = 0
        pruned_params = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in self.masks:
                mask = self.masks[name]
                total_params += mask.numel()
                pruned_params += (mask == 0).sum().item()
        current_sparsity = pruned_params / total_params if total_params > 0 else 0.0
        return {
            'total_parameters': total_params,
            'pruned_parameters': pruned_params,
            'remaining_parameters': total_params - pruned_params,
            'sparsity_ratio': current_sparsity,
            'mask_frozen': self.mask_frozen,
            'total_mask_updates': len(self.mask_update_history)
        }

# Data loading
print("Loading CIFAR-10...")
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normalize
])
test_transform = transforms.Compose([transforms.ToTensor(), normalize])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create model
print("\nCreating ResNet-56 model...")
model = ResNet56(num_classes=10).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-4, nesterov=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)

# Create MAP pruner (notebook implementation)
map_pruner = MAPPruner(model=model, target_sparsity=0.9, mask_update_freq=16,
                      total_epochs=50, exploration_end=37, exploitation_start=45)

criterion = nn.CrossEntropyLoss()

print("\nStarting training (Notebook Implementation - Sigmoid + Linear)...")
print("="*80)

best_test_acc = 0.0
start_time = time.time()

for epoch in range(50):
    # Train
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:2d}/50')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        if map_pruner is not None:
            map_pruner.step(epoch)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 50 == 0:
            current_acc = 100.0 * correct / total
            current_loss = total_loss / (batch_idx + 1)
            current_sparsity = map_pruner.current_sparsity if map_pruner else 0.0
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%', 'Sparsity': f'{current_sparsity:.3f}'})
    
    train_loss = total_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    
    # Evaluate
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_acc = 100.0 * correct / total
    test_loss = test_loss / len(test_loader)
    
    scheduler.step()
    
    stats = map_pruner.get_statistics()
    
    print(f"\nEpoch {epoch+1:2d}/50 Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
    print(f"  Sparsity: {stats['sparsity_ratio']:.4f} ({stats['sparsity_ratio']*100:.2f}%)")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        print(f"  ✓ New Best: {best_test_acc:.2f}%")

total_time = time.time() - start_time
print("\n" + "="*80)
print("TRAINING COMPLETE (NOTEBOOK IMPLEMENTATION)")
print("="*80)
print(f"Total Time: {total_time/60:.2f} minutes")
print(f"Best Test Accuracy: {best_test_acc:.2f}%")
print(f"Final Sparsity: {stats['sparsity_ratio']*100:.2f}%")
print(f"Total Mask Updates: {stats['total_mask_updates']}")
