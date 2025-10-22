"""
Pruning Methods for Neural Networks
Implements magnitude-based pruning techniques
"""

import torch
import torch.nn as nn
import numpy as np


class MagnitudePruning:
    """
    Basic one-shot magnitude-based pruning
    Prunes weights with the smallest absolute values
    """
    def __init__(self, model, prune_ratio=0.5):
        """
        Args:
            model: PyTorch model to prune
            prune_ratio: Fraction of weights to prune (0-1)
        """
        self.model = model
        self.prune_ratio = prune_ratio
        
    def prune(self):
        """Apply magnitude-based pruning to the model"""
        # Collect all weights
        all_weights = []
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad and len(param.shape) >= 2:
                all_weights.append(param.data.abs().view(-1))
        
        # Concatenate all weights
        all_weights = torch.cat(all_weights)
        
        # Find threshold
        k = int(self.prune_ratio * len(all_weights))
        if k > 0:
            threshold = torch.topk(all_weights, k, largest=False).values.max()
        else:
            threshold = 0
        
        # Apply pruning mask
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad and len(param.shape) >= 2:
                mask = (param.data.abs() > threshold).float()
                param.data *= mask
        
        return self.model


class IterativeMagnitudePruning:
    """
    Iterative magnitude-based pruning with fine-tuning
    Gradually prunes the network over multiple iterations
    """
    def __init__(self, model, trainloader, prune_ratio=0.5, iterations=5, 
                 epochs_per_iteration=5, lr=0.01, device='cuda'):
        """
        Args:
            model: PyTorch model to prune
            trainloader: DataLoader for training data
            prune_ratio: Final fraction of weights to prune (0-1)
            iterations: Number of pruning iterations
            epochs_per_iteration: Training epochs after each pruning step
            lr: Learning rate for fine-tuning
            device: Device to use for training
        """
        self.model = model
        self.trainloader = trainloader
        self.final_prune_ratio = prune_ratio
        self.iterations = iterations
        self.epochs_per_iteration = epochs_per_iteration
        self.lr = lr
        self.device = device
        
    def prune_and_finetune(self):
        """Apply iterative pruning with fine-tuning"""
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, 
                                   momentum=0.9, weight_decay=1e-4)
        
        # Calculate pruning ratio per iteration
        ratio_per_iter = 1 - (1 - self.final_prune_ratio) ** (1 / self.iterations)
        
        for iteration in range(self.iterations):
            print(f"\nIteration {iteration + 1}/{self.iterations}")
            print(f"Pruning ratio this iteration: {ratio_per_iter:.2%}")
            
            # Apply pruning
            pruner = MagnitudePruning(self.model, prune_ratio=ratio_per_iter)
            self.model = pruner.prune()
            
            # Fine-tune
            for epoch in range(self.epochs_per_iteration):
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for i, (inputs, targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                acc = 100. * correct / total
                print(f"  Epoch {epoch+1}/{self.epochs_per_iteration}: "
                      f"Loss: {running_loss/len(self.trainloader):.3f}, "
                      f"Acc: {acc:.2f}%")
        
        return self.model
