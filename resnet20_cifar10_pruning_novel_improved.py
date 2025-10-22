"""
ResNet-20 IMPROVED Novel MAP Pruning on CIFAR-10
Enhanced with fixes based on comparative analysis:
1. Post-training pruning (like traditional method)
2. Multi-criteria attention for mask selection only
3. Proper fine-tuning with frozen masks
4. Fixed phase scheduling and attention calibration

This version addresses all issues identified in the comparison study.
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
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
try:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using device: cuda:0")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"Using device: cpu (CUDA not available)")
except Exception as e:
    device = torch.device("cpu")
    print(f"Using device: cpu (CUDA error: {e})")


# ================================================================================================
# DATA LOADING
# ================================================================================================

def load_cifar10():
    """Load CIFAR-10 dataset with standard augmentation"""
    print("Loading CIFAR-10 dataset...")
    
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
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return trainloader, testloader


# ================================================================================================
# MODEL ARCHITECTURE
# ================================================================================================

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
        
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        
        self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)
        
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


# ================================================================================================
# ATTENTION MECHANISMS (IMPROVED & CALIBRATED)
# ================================================================================================

class AttentionHead:
    """Base class for attention mechanisms"""
    def __init__(self, name: str):
        self.name = name
        
    def compute_attention(self, param: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute attention scores for given parameters"""
        raise NotImplementedError


class MagnitudeAttention(AttentionHead):
    """Magnitude-based attention: A_m(w) = |w| / max(|w|)"""
    def __init__(self):
        super().__init__("magnitude")
        
    def compute_attention(self, param: torch.Tensor, **kwargs) -> torch.Tensor:
        abs_param = torch.abs(param)
        max_val = abs_param.max()
        if max_val > 0:
            return abs_param / max_val
        return torch.ones_like(abs_param)  # Changed from zeros to ones


class GradientAttention(AttentionHead):
    """
    IMPROVED Gradient-based attention with proper calibration
    Uses exponential moving average for stability
    """
    def __init__(self, momentum=0.95):  # Increased momentum for stability
        super().__init__("gradient")
        self.momentum = momentum
        self.running_avg_grad = {}
        self.initialized = {}
        
    def compute_attention(self, param: torch.Tensor, param_name: str = None, **kwargs) -> torch.Tensor:
        if param.grad is None:
            return torch.ones_like(param)  # Changed: no gradient = uniform attention
            
        abs_grad = torch.abs(param.grad).detach()
        
        # Initialize or update running average
        if param_name not in self.running_avg_grad:
            self.running_avg_grad[param_name] = abs_grad.clone()
            self.initialized[param_name] = True
        else:
            self.running_avg_grad[param_name] = (
                self.momentum * self.running_avg_grad[param_name] + 
                (1 - self.momentum) * abs_grad
            )
        
        avg_grad = self.running_avg_grad[param_name]
        max_val = avg_grad.max()
        
        if max_val > 1e-10:  # Added threshold to avoid division by very small numbers
            return avg_grad / max_val
        return torch.ones_like(param)


class EntropyAttention(AttentionHead):
    """
    IMPROVED Information-theoretic attention
    Simplified to use weight variance as proxy for information content
    """
    def __init__(self):
        super().__init__("entropy")
        
    def compute_attention(self, param: torch.Tensor, param_name: str = None, **kwargs) -> torch.Tensor:
        # Compute variance-based importance per filter/neuron
        # Higher variance = more diverse = more important
        
        if len(param.shape) == 4:  # Conv layers: [out_ch, in_ch, h, w]
            # Variance per output filter
            variance = param.view(param.shape[0], -1).var(dim=1, keepdim=True)
            # Broadcast to full shape
            attention = variance.unsqueeze(-1).unsqueeze(-1).expand_as(param)
        elif len(param.shape) == 2:  # Linear layers: [out, in]
            # Variance per output neuron
            variance = param.var(dim=1, keepdim=True)
            attention = variance.expand_as(param)
        else:
            return torch.ones_like(param)
        
        # Normalize
        max_val = attention.max()
        if max_val > 1e-10:
            return attention / max_val
        return torch.ones_like(param)


class HessianAttention(AttentionHead):
    """
    IMPROVED Second-order attention using Fisher Information
    Better approximation with stabilized accumulation
    """
    def __init__(self, momentum=0.95, damping=1e-5):
        super().__init__("hessian")
        self.momentum = momentum
        self.damping = damping
        self.fisher_info = {}
        
    def compute_attention(self, param: torch.Tensor, param_name: str = None, **kwargs) -> torch.Tensor:
        if param.grad is None:
            return torch.ones_like(param)
        
        # Fisher information: E[grad^2]
        squared_grad = (param.grad ** 2).detach()
        
        if param_name not in self.fisher_info:
            self.fisher_info[param_name] = squared_grad.clone()
        else:
            self.fisher_info[param_name] = (
                self.momentum * self.fisher_info[param_name] + 
                (1 - self.momentum) * squared_grad
            )
        
        fisher = self.fisher_info[param_name]
        
        # FIXED: Higher curvature = MORE important (should have higher attention)
        # Original was inverted - we want to KEEP high-curvature weights
        attention = fisher / (fisher.mean() + self.damping)
        
        # Normalize
        max_val = attention.max()
        if max_val > 1e-10:
            return attention / max_val
        return torch.ones_like(param)


# ================================================================================================
# IMPROVED NOVEL MAP PRUNER
# ================================================================================================

class ImprovedNovelMAPPruner:
    """
    Improved Multi-Criteria Attention-based Pruning
    
    KEY CHANGES:
    1. Post-training pruning (not progressive)
    2. Multi-criteria attention for mask selection ONLY
    3. Separate fine-tuning phase with frozen masks
    4. Fixed attention head calibration
    5. Proper fusion weight configuration
    """
    def __init__(self, model, trainloader, testloader, prune_ratio=0.5, 
                 fusion_mode='balanced', device='cuda'):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.prune_ratio = prune_ratio
        self.device = device
        self.fusion_mode = fusion_mode
        
        # Initialize attention heads with improved implementations
        self.attention_heads = {
            'magnitude': MagnitudeAttention(),
            'gradient': GradientAttention(momentum=0.95),
            'entropy': EntropyAttention(),
            'hessian': HessianAttention(momentum=0.95)
        }
        
        # IMPROVED fusion weights based on analysis
        self.fusion_weights = self._get_fusion_weights(fusion_mode)
        
        # Store masks
        self.masks = {}
        
    def _get_fusion_weights(self, mode='balanced'):
        """
        Get fusion weights for different strategies
        """
        if mode == 'magnitude_only':
            return {'magnitude': 1.0, 'gradient': 0.0, 'entropy': 0.0, 'hessian': 0.0}
        elif mode == 'balanced':
            return {'magnitude': 0.4, 'gradient': 0.3, 'entropy': 0.2, 'hessian': 0.1}
        elif mode == 'gradient_focused':
            return {'magnitude': 0.3, 'gradient': 0.5, 'entropy': 0.1, 'hessian': 0.1}
        elif mode == 'multi_criteria':
            return {'magnitude': 0.25, 'gradient': 0.25, 'entropy': 0.25, 'hessian': 0.25}
        else:
            return {'magnitude': 0.4, 'gradient': 0.3, 'entropy': 0.2, 'hessian': 0.1}
    
    def get_prunable_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """Get list of prunable parameters"""
        prunable = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                if 'weight' in name and ('conv' in name or 'linear' in name):
                    prunable.append((name, param))
        return prunable
    
    def collect_gradients(self, num_batches=100):
        """
        Collect gradients for gradient and Hessian attention
        Run a few forward-backward passes to accumulate gradient statistics
        """
        print(f"Collecting gradient statistics over {num_batches} batches...")
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        batch_count = 0
        for inputs, targets in self.trainloader:
            if batch_count >= num_batches:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update attention head statistics (they accumulate internally)
            for param_name, param in self.get_prunable_parameters():
                self.attention_heads['gradient'].compute_attention(param, param_name=param_name)
                self.attention_heads['hessian'].compute_attention(param, param_name=param_name)
            
            batch_count += 1
            
            if batch_count % 20 == 0:
                print(f"  Processed {batch_count}/{num_batches} batches")
        
        print("Gradient collection complete!")
    
    def compute_combined_attention(self, param: torch.Tensor, param_name: str) -> torch.Tensor:
        """
        Compute combined attention score using all attention heads
        """
        # Compute attention from each head
        attention_scores = {}
        for head_name, head in self.attention_heads.items():
            attention_scores[head_name] = head.compute_attention(param, param_name=param_name)
        
        # Combine using fusion weights
        combined = torch.zeros_like(param)
        total_weight = sum(self.fusion_weights.values())
        
        for head_name, weight in self.fusion_weights.items():
            if head_name in attention_scores and weight > 0:
                combined += (weight / total_weight) * attention_scores[head_name]
        
        return combined
    
    def create_pruning_mask(self, param: torch.Tensor, attention_scores: torch.Tensor,
                           prune_ratio: float) -> torch.Tensor:
        """
        Create pruning mask based on attention scores
        Higher attention = more important = KEEP (mask = 1)
        Lower attention = less important = PRUNE (mask = 0)
        """
        attention_flat = attention_scores.view(-1)
        
        # Number of weights to prune
        k = int(prune_ratio * len(attention_flat))
        
        if k > 0:
            # Find threshold: prune k weights with LOWEST attention
            threshold = torch.topk(attention_flat, k, largest=False).values.max()
            mask = (attention_flat > threshold).float()
        else:
            mask = torch.ones_like(attention_flat)
        
        return mask.view(param.shape)
    
    def apply_pruning(self):
        """
        Apply multi-criteria attention-based pruning
        This is called AFTER initial training (like traditional method)
        """
        print(f"\n{'='*80}")
        print(f"Applying Improved Novel MAP Pruning")
        print(f"Fusion Mode: {self.fusion_mode}")
        print(f"Fusion Weights: {self.fusion_weights}")
        print(f"{'='*80}\n")
        
        prunable_params = self.get_prunable_parameters()
        
        for param_name, param in prunable_params:
            # Compute combined attention
            attention_scores = self.compute_combined_attention(param, param_name)
            
            # Create pruning mask
            mask = self.create_pruning_mask(param, attention_scores, self.prune_ratio)
            
            # Store mask
            self.masks[param_name] = mask
            
            # Apply mask to parameter
            with torch.no_grad():
                param.data *= mask
            
            # Calculate sparsity for this layer
            sparsity = 1.0 - (mask.sum() / mask.numel()).item()
            print(f"  {param_name}: shape={list(param.shape)}, sparsity={sparsity*100:.2f}%")
        
        # Calculate global sparsity
        total_params = sum(p.numel() for _, p in prunable_params)
        nonzero_params = sum((self.masks[n] != 0).sum().item() for n, _ in prunable_params)
        global_sparsity = 1.0 - (nonzero_params / total_params)
        
        print(f"\nGlobal sparsity: {global_sparsity*100:.2f}%")
        print(f"{'='*80}\n")
        
        return global_sparsity
    
    def enforce_masks(self):
        """Enforce pruning masks during fine-tuning"""
        for param_name, mask in self.masks.items():
            for name, param in self.model.named_parameters():
                if name == param_name:
                    with torch.no_grad():
                        param.data *= mask
                    break


# ================================================================================================
# TRAINING AND EVALUATION
# ================================================================================================

def train_model(model, trainloader, epochs=30, lr=0.1):
    """Train the ResNet-20 model (standard training)"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
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
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if i % 100 == 99:
                print(f'  [Epoch {epoch+1}/{epochs}, Batch {i+1}] '
                      f'Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        train_acc = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} completed - Train Accuracy: {train_acc:.2f}%, '
              f'LR: {scheduler.get_last_lr()[0]:.4f}')
        scheduler.step()
    
    print('Finished Training')
    return model


def fine_tune_pruned_model(model, trainloader, pruner, epochs=10, lr=0.01):
    """
    Fine-tune pruned model with frozen masks (like traditional method)
    This is the KEY difference - proper fine-tuning AFTER pruning
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    milestones = [int(epochs * 0.5), int(epochs * 0.75)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    print(f"Fine-tuning for {epochs} epochs with lr={lr}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # CRITICAL: Enforce masks after each update
            pruner.enforce_masks()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if i % 100 == 99:
                print(f'  [Epoch {epoch+1}/{epochs}, Batch {i+1}] '
                      f'Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        train_acc = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}%, '
              f'LR: {scheduler.get_last_lr()[0]:.4f}')
        scheduler.step()
    
    print('Finished Fine-tuning')
    return model


def evaluate_model(model, testloader, verbose=True):
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
    if verbose:
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


def plot_comparison(traditional_results, novel_results, improved_results):
    """Plot comparison between all three methods"""
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ratios = [r['ratio'] for r in improved_results]
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0, 0]
    trad_acc = [r['accuracy'] for r in traditional_results]
    novel_acc = [r['accuracy'] for r in novel_results]
    improved_acc = [r['accuracy'] for r in improved_results]
    
    x = np.arange(len(ratios))
    width = 0.25
    
    ax1.bar(x - width, trad_acc, width, label='Traditional', color='blue', alpha=0.7)
    ax1.bar(x, novel_acc, width, label='Novel MAP', color='red', alpha=0.7)
    ax1.bar(x + width, improved_acc, width, label='Improved MAP', color='green', alpha=0.7)
    
    ax1.set_xlabel('Pruning Ratio')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{int(r*100)}%' for r in ratios])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Sparsity vs Accuracy
    ax2 = axes[0, 1]
    trad_sparsity = [r['sparsity'] * 100 for r in traditional_results]
    novel_sparsity = [r['sparsity'] * 100 for r in novel_results]
    improved_sparsity = [r['sparsity'] * 100 for r in improved_results]
    
    ax2.plot(trad_sparsity, trad_acc, 'bo-', linewidth=2, markersize=8, label='Traditional')
    ax2.plot(novel_sparsity, novel_acc, 'ro-', linewidth=2, markersize=8, label='Novel MAP')
    ax2.plot(improved_sparsity, improved_acc, 'go-', linewidth=2, markersize=8, label='Improved MAP')
    
    ax2.set_xlabel('Sparsity (%)')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Accuracy vs Sparsity Trade-off')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy drop comparison
    ax3 = axes[1, 0]
    trad_drop = [traditional_results[0]['accuracy'] - r['accuracy'] for r in traditional_results]
    novel_drop = [novel_results[0]['accuracy'] - r['accuracy'] for r in novel_results]
    improved_drop = [improved_results[0]['accuracy'] - r['accuracy'] for r in improved_results]
    
    ax3.bar(x - width, trad_drop, width, label='Traditional', color='blue', alpha=0.7)
    ax3.bar(x, novel_drop, width, label='Novel MAP', color='red', alpha=0.7)
    ax3.bar(x + width, improved_drop, width, label='Improved MAP', color='green', alpha=0.7)
    
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Pruning Ratio')
    ax3.set_ylabel('Accuracy Drop (%)')
    ax3.set_title('Accuracy Drop from Original')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{int(r*100)}%' for r in ratios])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [['Ratio', 'Traditional', 'Novel', 'Improved', 'Winner']]
    for i, r in enumerate(ratios):
        ratio_str = f"{int(r*100)}%"
        trad = f"{trad_acc[i]:.1f}%"
        novel = f"{novel_acc[i]:.1f}%"
        improved = f"{improved_acc[i]:.1f}%"
        
        # Determine winner
        best = max(trad_acc[i], novel_acc[i], improved_acc[i])
        if improved_acc[i] == best:
            winner = "Improved ✓"
        elif trad_acc[i] == best:
            winner = "Trad ✓"
        else:
            winner = "Novel ✓"
        
        table_data.append([ratio_str, trad, novel, improved, winner])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Three-Way Comparison: Traditional vs Novel vs Improved MAP', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/improved_map_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved to plots/improved_map_comparison.png")
    plt.close()


# ================================================================================================
# MAIN EXPERIMENT
# ================================================================================================

def main():
    print("="*80)
    print("IMPROVED Novel MAP Pruning for ResNet-20")
    print("="*80)
    
    # Load CIFAR-10 dataset
    trainloader, testloader = load_cifar10()
    
    # Load the pre-trained original model
    print("\nLoading pre-trained original model...")
    original_model = ResNet20(num_classes=10)
    
    # Check if we have a saved original model
    if os.path.exists('models/resnet20_cifar10_original.pth'):
        print("Loading from saved checkpoint...")
        original_model.load_state_dict(torch.load('models/resnet20_cifar10_original.pth'))
        original_model.to(device)
    else:
        print("Training original model from scratch...")
        original_model.to(device)
        original_model = train_model(original_model, trainloader, epochs=30, lr=0.1)
        save_model(original_model, "resnet20_cifar10_original_improved.pth")
    
    # Evaluate original model
    print("\n" + "="*80)
    print("EVALUATING ORIGINAL MODEL")
    print("="*80)
    original_accuracy = evaluate_model(original_model, testloader)
    
    # Test different fusion strategies
    fusion_modes = ['balanced', 'magnitude_only', 'gradient_focused', 'multi_criteria']
    
    pruning_ratios = [0.3, 0.5, 0.7, 0.9]
    all_results = {}
    
    for fusion_mode in fusion_modes:
        print(f"\n{'='*80}")
        print(f"TESTING FUSION MODE: {fusion_mode.upper()}")
        print(f"{'='*80}")
        
        results = []
        
        for ratio in pruning_ratios:
            print(f"\n{'-'*80}")
            print(f"Pruning with ratio: {ratio} ({ratio*100:.0f}%), Mode: {fusion_mode}")
            print(f"{'-'*80}")
            
            # Create a copy of the trained model
            pruned_model = ResNet20(num_classes=10)
            pruned_model.load_state_dict(original_model.state_dict())
            pruned_model.to(device)
            
            # Initialize pruner
            pruner = ImprovedNovelMAPPruner(
                model=pruned_model,
                trainloader=trainloader,
                testloader=testloader,
                prune_ratio=ratio,
                fusion_mode=fusion_mode,
                device=device
            )
            
            # Collect gradient statistics
            pruner.collect_gradients(num_batches=50)
            
            # Apply pruning (POST-TRAINING, like traditional method)
            global_sparsity = pruner.apply_pruning()
            
            # Evaluate before fine-tuning
            print("\nEvaluating before fine-tuning...")
            acc_before = evaluate_model(pruned_model, testloader)
            
            # Fine-tune with frozen masks (KEY IMPROVEMENT)
            print("\nFine-tuning with frozen masks...")
            pruned_model = fine_tune_pruned_model(
                pruned_model, trainloader, pruner, epochs=10, lr=0.01
            )
            
            # Final evaluation
            print("\nFinal evaluation:")
            final_accuracy = evaluate_model(pruned_model, testloader)
            accuracy_drop = original_accuracy - final_accuracy
            
            # Get final sparsity
            total_params, nonzero_params = count_parameters(pruned_model)
            final_sparsity = 1 - (nonzero_params / total_params)
            
            print(f"\nFinal Results:")
            print(f"  Sparsity: {final_sparsity*100:.2f}%")
            print(f"  Accuracy before fine-tuning: {acc_before:.2f}%")
            print(f"  Accuracy after fine-tuning: {final_accuracy:.2f}%")
            print(f"  Improvement from fine-tuning: {final_accuracy - acc_before:.2f}%")
            print(f"  Accuracy drop from original: {accuracy_drop:.2f}%")
            
            # Save model
            save_model(pruned_model, 
                      f"resnet20_cifar10_improved_map_{fusion_mode}_{int(ratio*100)}.pth")
            
            results.append({
                'ratio': ratio,
                'sparsity': final_sparsity,
                'accuracy_before': acc_before,
                'accuracy': final_accuracy,
                'improvement': final_accuracy - acc_before,
                'drop': accuracy_drop
            })
        
        all_results[fusion_mode] = results
    
    # Print summary for best fusion mode (balanced)
    print("\n" + "="*80)
    print("SUMMARY - BALANCED FUSION MODE")
    print("="*80)
    print(f"\nOriginal Model Accuracy: {original_accuracy:.2f}%\n")
    print(f"{'Ratio':<10} {'Sparsity':<12} {'Acc Before':<12} {'Acc After':<12} {'Improvement':<12} {'Drop':<10}")
    print("-" * 75)
    
    for result in all_results['balanced']:
        print(f"{result['ratio']*100:>5.0f}%{'':<5} {result['sparsity']*100:>6.2f}%{'':<6} "
              f"{result['accuracy_before']:>6.2f}%{'':<6} {result['accuracy']:>6.2f}%{'':<6} "
              f"{result['improvement']:>+6.2f}%{'':<6} {result['drop']:>5.2f}%")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
