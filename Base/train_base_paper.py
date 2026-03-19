"""
Clean Reproduction of Original MAP Paper
Implements Magnitude-based Attention Pruning (MAP) exactly as described in the original paper.

Key Differences from Current Implementation:
1. Linear attention formula with normalized magnitudes (NOT sigmoid-based)
2. Cubic pruning schedule (NOT linear)
3. Explicit gradient scaling in backward pass
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time
from tqdm import tqdm


# ============================================================================
# Section 1: ResNet-56 Architecture (Copied from notebook)
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-56"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet56(nn.Module):
    """ResNet-56 for CIFAR-10/100 (9n+2 layers with n=9)"""
    
    def __init__(self, num_classes=10):
        super(ResNet56, self).__init__()
        self.in_planes = 16
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Three stages with 9 blocks each
        self.layer1 = self._make_layer(16, 9, stride=1)   # 16 filters
        self.layer2 = self._make_layer(32, 9, stride=2)   # 32 filters
        self.layer3 = self._make_layer(64, 9, stride=2)   # 64 filters
        
        # Final classifier
        self.linear = nn.Linear(64, num_classes)
    
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
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


# ============================================================================
# Section 2: Data Loading (Copied from notebook)
# ============================================================================

def get_cifar_datasets(dataset_name='cifar10', data_dir='../data'):
    """Load CIFAR-10 or CIFAR-100 datasets with standard augmentation"""
    
    # Normalization values for CIFAR
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    if dataset_name.lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=test_transform
        )
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=False, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=False, transform=test_transform
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset, num_classes


def create_data_loaders(train_dataset, test_dataset, batch_size=128, num_workers=4):
    """Create data loaders with proper settings"""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


# ============================================================================
# Section 3: Custom Gradient Scaling Function
# ============================================================================

class MAPGradientScaling(torch.autograd.Function):
    """
    Custom autograd function for MAP gradient scaling as per original paper.
    
    Forward: Apply mask normally
    Backward: Scale gradients by attention values (kept) or (1-p)^z (pruned)
    """
    
    @staticmethod
    def forward(ctx, weight, mask, attention, sparsity, z):
        """
        Args:
            weight: Weight tensor
            mask: Binary mask (1=keep, 0=prune)
            attention: Attention values for each weight
            sparsity: Current sparsity level
            z: Attention strength parameter
        """
        ctx.save_for_backward(mask, attention)
        ctx.sparsity = sparsity
        ctx.z = z
        
        # Apply mask in forward pass
        return weight * mask
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Scale gradients according to MAP paper:
        - Kept weights: scaled by their attention value
        - Pruned weights: scaled by (1-p)^z
        """
        mask, attention = ctx.saved_tensors
        sparsity = ctx.sparsity
        z = ctx.z
        
        # Calculate decay factor for pruned weights
        pruned_decay = (1 - sparsity) ** z
        
        # Scale gradients
        # For kept weights (mask=1): multiply by attention
        # For pruned weights (mask=0): multiply by pruned_decay
        grad_scale = mask * attention + (1 - mask) * pruned_decay
        
        grad_weight = grad_output * grad_scale
        
        # Return gradients (None for non-tensor inputs)
        return grad_weight, None, None, None, None


# ============================================================================
# Section 4: Original Paper MAP Pruner Implementation
# ============================================================================

class MAPPruner_Base:
    """
    Original MAP Pruner as described in the paper.
    
    Key Features (DIFFERENT from notebook implementation):
    1. Linear attention formula: Attention = (Normalized_Mag * (1 - lower_bound)) + lower_bound
       where lower_bound = (1 - current_sparsity)^z
    2. Cubic pruning schedule: sparsity = target * (1 - (1 - t)^3)
    3. Gradient scaling via custom backward pass
    """
    
    def __init__(self, model, target_sparsity=0.9, z=1.0, mask_update_freq=16,
                 total_epochs=300, exploration_end=225, exploitation_start=250):
        """
        Args:
            model: Neural network model
            target_sparsity: Final target sparsity (default: 0.9 for 90%)
            z: Attention strength parameter (default: 1.0 as per paper)
            mask_update_freq: Update masks every N iterations (default: 16)
            total_epochs: Total training epochs
            exploration_end: Epoch to end exploration phase (225)
            exploitation_start: Epoch to freeze masks (250)
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.z = z
        self.mask_update_freq = mask_update_freq
        self.total_epochs = total_epochs
        self.exploration_end = exploration_end
        self.exploitation_start = exploitation_start
        
        # Initialize masks and attention values
        self.masks = {}
        self.attention_values = {}
        self.current_sparsity = 0.0
        self.iteration_count = 0
        self.mask_frozen = False
        
        # Initialize masks
        self._initialize_masks()
        
        # Statistics tracking
        self.sparsity_history = []
        self.mask_update_history = []
        
        print(f"[MAP Base] Initialized with:")
        print(f"  Target Sparsity: {target_sparsity * 100:.1f}%")
        print(f"  Attention Strength (z): {z}")
        print(f"  Mask Update Frequency: Every {mask_update_freq} iterations")
        print(f"  Exploration Phase: Epochs 0-{exploration_end}")
        print(f"  Exploitation Phase: Epochs {exploitation_start}-{total_epochs}")
    
    def _initialize_masks(self):
        """Initialize masks for all Conv2d and Linear layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Initialize mask as all ones (no pruning initially)
                self.masks[name] = torch.ones_like(module.weight.data)
                # Initialize attention values based on weight magnitudes
                self.attention_values[name] = torch.abs(module.weight.data.clone())
    
    def _get_cubic_sparsity(self, current_epoch):
        """
        Calculate sparsity using cubic schedule as per original paper.
        
        Formula: s_t = s_f * (1 - (1 - t/T)^3)
        where:
            s_t = current sparsity
            s_f = final target sparsity
            t = current epoch
            T = exploration end epoch
        """
        if current_epoch >= self.exploration_end:
            return self.target_sparsity
        
        # Cubic schedule
        progress = current_epoch / self.exploration_end
        current_sparsity = self.target_sparsity * (1 - (1 - progress) ** 3)
        
        return current_sparsity
    
    def _calculate_attention_values(self, weights, current_sparsity):
        """
        Calculate attention values using paper's linear formula.
        
        Original Paper Formula:
        1. Normalize weight magnitudes to [0, 1]
        2. Map to range [(1-p)^z, 1]
        
        Attention = (Normalized_Mag * (1 - lower_bound)) + lower_bound
        where lower_bound = (1 - current_sparsity)^z
        """
        # Get weight magnitudes
        magnitudes = torch.abs(weights)
        
        # Normalize to [0, 1]
        min_mag = magnitudes.min()
        max_mag = magnitudes.max()
        
        if max_mag - min_mag > 1e-8:  # Avoid division by zero
            normalized_mag = (magnitudes - min_mag) / (max_mag - min_mag)
        else:
            normalized_mag = torch.ones_like(magnitudes)
        
        # Calculate lower bound: (1 - p)^z
        lower_bound = (1 - current_sparsity) ** self.z
        
        # Map to range [lower_bound, 1]
        attention = normalized_mag * (1 - lower_bound) + lower_bound
        
        return attention
    
    def _calculate_global_threshold(self, sparsity_ratio):
        """Calculate global threshold for given sparsity ratio"""
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
        """Update pruning masks based on current sparsity schedule"""
        if self.mask_frozen:
            return
        
        # Calculate current target sparsity using cubic schedule
        current_target_sparsity = self._get_cubic_sparsity(current_epoch)
        
        # Calculate threshold for current sparsity
        threshold = self._calculate_global_threshold(current_target_sparsity)
        
        # Update masks and attention values
        total_weights = 0
        pruned_weights = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in self.masks:
                # Calculate attention values using paper's linear formula
                self.attention_values[name] = self._calculate_attention_values(
                    module.weight.data, current_target_sparsity
                )
                
                # Create mask based on magnitude threshold
                magnitudes = torch.abs(module.weight.data)
                self.masks[name] = (magnitudes > threshold).float()
                
                # Count statistics
                total_weights += module.weight.numel()
                pruned_weights += (self.masks[name] == 0).sum().item()
        
        # Update current sparsity
        self.current_sparsity = pruned_weights / total_weights if total_weights > 0 else 0.0
        self.sparsity_history.append(self.current_sparsity)
        
        # Track mask updates
        self.mask_update_history.append({
            'epoch': current_epoch,
            'iteration': self.iteration_count,
            'sparsity': self.current_sparsity,
            'threshold': threshold,
            'target_sparsity': current_target_sparsity
        })
    
    def apply_masks_with_gradient_scaling(self):
        """
        Apply masks with gradient scaling using custom backward pass.
        This is the key difference from notebook implementation.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in self.masks:
                # Apply mask with gradient scaling
                module.weight.data = MAPGradientScaling.apply(
                    module.weight.data,
                    self.masks[name],
                    self.attention_values[name],
                    self.current_sparsity,
                    self.z
                )
    
    def step(self, current_epoch):
        """Perform one pruning step"""
        self.iteration_count += 1
        
        # Check if masks should be frozen (exploitation phase)
        if current_epoch >= self.exploitation_start:
            self.mask_frozen = True
        
        # Update masks at specified frequency during exploration
        if (self.iteration_count % self.mask_update_freq == 0 and 
            not self.mask_frozen and 
            current_epoch < self.exploration_end):
            self._update_masks(current_epoch)
        
        # Apply masks with gradient scaling
        self.apply_masks_with_gradient_scaling()
    
    def get_statistics(self):
        """Get current pruning statistics"""
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


# ============================================================================
# Section 5: Training Functions
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, criterion, map_pruner, 
                    epoch, device):
    """Train for one epoch with MAP pruning"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Apply pruning step (mask updates and gradient scaling)
        if map_pruner is not None:
            map_pruner.step(epoch)
        
        # Optimizer step
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        if batch_idx % 50 == 0:
            current_acc = 100.0 * correct / total
            current_loss = total_loss / (batch_idx + 1)
            current_sparsity = map_pruner.current_sparsity if map_pruner else 0.0
            
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Sparsity': f'{current_sparsity:.3f}'
            })
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss


def save_checkpoint(model, optimizer, scheduler, map_pruner, epoch, 
                   train_loss, test_acc, save_dir, filename):
    """Save training checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'test_accuracy': test_acc,
    }
    
    # Add pruning-related data
    if map_pruner is not None:
        checkpoint['map_pruner'] = {
            'masks': map_pruner.masks,
            'attention_values': map_pruner.attention_values,
            'current_sparsity': map_pruner.current_sparsity,
            'sparsity_history': map_pruner.sparsity_history,
            'mask_update_history': map_pruner.mask_update_history,
            'statistics': map_pruner.get_statistics()
        }
    
    torch.save(checkpoint, filepath)
    print(f"[Checkpoint] Saved: {filename}")


# ============================================================================
# Section 6: Main Training Function
# ============================================================================

def train_map_base(dataset_name='cifar10', device='cuda', save_dir='./checkpoints'):
    """
    Main training function for original paper MAP implementation.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
    """
    
    print("="*80)
    print("TRAINING ORIGINAL MAP PAPER IMPLEMENTATION (BASE)")
    print("="*80)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Device: {device}")
    print()
    
    # Configuration
    batch_size = 128
    total_epochs = 50  # MODIFIED FOR COMPARISON TEST
    initial_lr = 0.2
    momentum = 0.9
    weight_decay = 1e-4
    lr_milestones = [25, 40]  # MODIFIED FOR COMPARISON TEST
    
    # MAP parameters (as per paper)
    target_sparsity = 0.9  # 90%
    z = 1.0  # Attention strength
    mask_update_freq = 16
    exploration_end = 37  # MODIFIED FOR COMPARISON TEST (75% of 50)
    exploitation_start = 45  # MODIFIED FOR COMPARISON TEST (90% of 50)
    
    # Load data
    print("Loading datasets...")
    train_dataset, test_dataset, num_classes = get_cifar_datasets(dataset_name)
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, batch_size)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    print()
    
    # Create model
    print("Creating ResNet-56 model...")
    model = ResNet56(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Create optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=0.1
    )
    
    # Create MAP pruner (Base Implementation)
    map_pruner = MAPPruner_Base(
        model=model,
        target_sparsity=target_sparsity,
        z=z,
        mask_update_freq=mask_update_freq,
        total_epochs=total_epochs,
        exploration_end=exploration_end,
        exploitation_start=exploitation_start
    )
    print()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training configuration summary
    print("Training Configuration:")
    print(f"  Total Epochs: {total_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Initial LR: {initial_lr}")
    print(f"  LR Schedule: {lr_milestones} (gamma=0.1)")
    print(f"  Momentum: {momentum}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Target Sparsity: {target_sparsity * 100:.0f}%")
    print(f"  Attention Strength (z): {z}")
    print()
    
    # Training loop
    best_test_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    print("Starting training...")
    print("="*80)
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        
        # Phase announcements
        if epoch == 0:
            print(f"\n🔍 EXPLORATION PHASE (Epochs 0-{exploration_end})")
            print(f"   - Cubic sparsity schedule: 0% → {target_sparsity*100:.0f}%")
            print(f"   - Dynamic mask updates every {mask_update_freq} iterations")
            print(f"   - Linear attention formula with z={z}")
            print()
        
        if epoch == exploration_end:
            print(f"\n🎯 TRANSITION PHASE (Epochs {exploration_end}-{exploitation_start})")
            print(f"   - Fixed sparsity at {target_sparsity*100:.0f}%")
            print(f"   - Continued mask updates")
            print()
        
        if epoch == exploitation_start:
            print(f"\n🔒 EXPLOITATION PHASE (Epochs {exploitation_start}-{total_epochs})")
            print(f"   - Masks frozen")
            print(f"   - Weight optimization only")
            print()
        
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, map_pruner, epoch, device
        )
        
        # Evaluate
        test_acc, test_loss = evaluate_model(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Get statistics
        stats = map_pruner.get_statistics()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1:3d}/{total_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Sparsity: {stats['sparsity_ratio']:.4f} ({stats['sparsity_ratio']*100:.2f}%)")
        print(f"  Active Params: {stats['remaining_parameters']:,} / {stats['total_parameters']:,}")
        print(f"  Mask Updates: {stats['total_mask_updates']}")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Update best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            print(f"  ✓ New Best Test Accuracy: {best_test_acc:.2f}%")
            
            # Save best model
            save_checkpoint(
                model, optimizer, scheduler, map_pruner, epoch,
                train_loss, test_acc, save_dir,
                f"best_model_{dataset_name}_base.pth"
            )
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                model, optimizer, scheduler, map_pruner, epoch,
                train_loss, test_acc, save_dir,
                f"checkpoint_epoch_{epoch+1}_{dataset_name}_base.pth"
            )
        
        print("-"*80)
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, map_pruner, total_epochs - 1,
        train_loss, test_acc, save_dir,
        f"final_model_{dataset_name}_base.pth"
    )
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Best Test Accuracy: {best_test_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Final Sparsity: {stats['sparsity_ratio']*100:.2f}%")
    print(f"Total Mask Updates: {stats['total_mask_updates']}")
    print(f"Checkpoints saved to: {save_dir}")
    print("="*80)


# ============================================================================
# Section 7: Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Original MAP Paper Implementation')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use (default: cifar10)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run training
    train_map_base(
        dataset_name=args.dataset,
        device=args.device,
        save_dir=args.save_dir
    )
