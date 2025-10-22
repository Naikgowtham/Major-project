"""
Results Viewer and Comparison
View and compare results from both pruning methods
"""

import os
import glob
import torch
import matplotlib.pyplot as plt
from PIL import Image


def count_model_parameters(model_path):
    """Count parameters in a saved model"""
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        total = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
        nonzero = sum((p != 0).sum().item() for p in state_dict.values() if hasattr(p, 'numel'))
        return total, nonzero
    except:
        return None, None


def main():
    print("="*80)
    print("PRUNING METHODS COMPARISON - RESULTS VIEWER")
    print("="*80)
    
    # Check for models
    print("\n1. SAVED MODELS")
    print("-"*80)
    
    traditional_models = sorted(glob.glob("models/resnet20_cifar10_pruned_*.pth"))
    novel_models = sorted(glob.glob("models/resnet20_cifar10_novel_map_*.pth"))
    
    print(f"\nTraditional Magnitude Pruning Models: {len(traditional_models)}")
    for model in traditional_models:
        size = os.path.getsize(model) / (1024 * 1024)  # MB
        print(f"  - {os.path.basename(model)}: {size:.2f} MB")
    
    print(f"\nNovel MAP Pruning Models: {len(novel_models)}")
    for model in novel_models:
        size = os.path.getsize(model) / (1024 * 1024)  # MB
        print(f"  - {os.path.basename(model)}: {size:.2f} MB")
    
    # Check for plots
    print("\n\n2. GENERATED PLOTS")
    print("-"*80)
    
    plot_files = [
        "plots/pruning_results.png",
        "plots/novel_map_pruning_results.png"
    ]
    
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            size = os.path.getsize(plot_file) / 1024  # KB
            print(f"  ✓ {plot_file} ({size:.1f} KB)")
        else:
            print(f"  ✗ {plot_file} (not found)")
    
    # Show comparison
    print("\n\n3. QUICK COMPARISON")
    print("-"*80)
    
    comparison_data = {
        'Traditional': [],
        'Novel MAP': []
    }
    
    # Parse traditional models
    for model_path in traditional_models:
        ratio = os.path.basename(model_path).replace('resnet20_cifar10_pruned_', '').replace('.pth', '')
        total, nonzero = count_model_parameters(model_path)
        if total:
            sparsity = (1 - nonzero/total) * 100
            comparison_data['Traditional'].append({
                'ratio': ratio,
                'sparsity': sparsity,
                'params': nonzero
            })
    
    # Parse novel models
    for model_path in novel_models:
        ratio = os.path.basename(model_path).replace('resnet20_cifar10_novel_map_', '').replace('.pth', '')
        total, nonzero = count_model_parameters(model_path)
        if total:
            sparsity = (1 - nonzero/total) * 100
            comparison_data['Novel MAP'].append({
                'ratio': ratio,
                'sparsity': sparsity,
                'params': nonzero
            })
    
    print("\n Method          | Ratio | Sparsity | Non-zero Params")
    print("-" * 60)
    
    for method, models in comparison_data.items():
        for m in models:
            print(f" {method:15s} | {m['ratio']:5s} | {m['sparsity']:7.2f}% | {m['params']:>15,}")
    
    # View plots
    print("\n\n4. VIEW PLOTS")
    print("-"*80)
    print("\nTo view the plots, open:")
    print("  - plots/pruning_results.png (Traditional)")
    print("  - plots/novel_map_pruning_results.png (Novel MAP)")
    print("\nOr run: xdg-open plots/*.png")
    
    # Check for training logs
    print("\n\n5. TRAINING LOGS")
    print("-"*80)
    
    if os.path.exists("comparison_log.txt"):
        size = os.path.getsize("comparison_log.txt") / 1024  # KB
        print(f"  ✓ comparison_log.txt ({size:.1f} KB)")
        print("\n  View with: cat comparison_log.txt | less")
    else:
        print("  ✗ comparison_log.txt (not found)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
