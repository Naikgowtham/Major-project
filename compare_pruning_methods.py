"""
Comparison Script: Traditional Magnitude Pruning vs Novel MAP Pruning
This script runs both methods sequentially and generates a comprehensive comparison.
"""

import subprocess
import json
import os
import time
from datetime import datetime

def run_script(script_name, method_name):
    """Run a pruning script and capture results"""
    print(f"\n{'='*80}")
    print(f"Running {method_name}")
    print(f"Script: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Run the script
    result = subprocess.run(
        f"source venv/bin/activate && python {script_name}",
        shell=True,
        cwd="/home/gowtham/7th sem/MP/Project",
        executable='/bin/bash',
        capture_output=False,  # Show output in real-time
        text=True
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"{method_name} Completed")
    print(f"Duration: {duration/60:.2f} minutes")
    print(f"Exit code: {result.returncode}")
    print(f"{'='*80}\n")
    
    return {
        'method': method_name,
        'script': script_name,
        'duration': duration,
        'exit_code': result.returncode
    }

def main():
    print("="*80)
    print("COMPARATIVE STUDY: Magnitude Pruning vs Novel MAP Pruning")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script will run both pruning methods sequentially:")
    print("1. Traditional Magnitude-Based Pruning")
    print("2. Novel MAP (Multi-Criteria Attention) Pruning")
    print("\nBoth will use identical parameters:")
    print("  - Initial training: 30 epochs")
    print("  - Fine-tuning: 10 epochs")
    print("  - Pruning ratios: [0.3, 0.5, 0.7, 0.9]")
    print("="*80)
    
    results = []
    
    # Run traditional magnitude pruning
    result1 = run_script("resnet20_cifar10_pruning.py", "Traditional Magnitude Pruning")
    results.append(result1)
    
    # Wait a bit between runs
    print("\nWaiting 10 seconds before starting next experiment...\n")
    time.sleep(10)
    
    # Run novel MAP pruning
    result2 = run_script("resnet20_cifar10_pruning_novel.py", "Novel MAP Pruning")
    results.append(result2)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nTotal experiment time: {(results[0]['duration'] + results[1]['duration'])/60:.2f} minutes")
    print("\nExecution Details:")
    for r in results:
        print(f"\n{r['method']}:")
        print(f"  Duration: {r['duration']/60:.2f} minutes")
        print(f"  Exit code: {r['exit_code']}")
    
    print("\n" + "="*80)
    print("RESULTS LOCATION:")
    print("="*80)
    print("\nModels saved in: ./models/")
    print("  - Traditional: resnet20_cifar10_pruned_*.pth")
    print("  - Novel MAP: resnet20_cifar10_novel_map_*.pth")
    print("\nPlots saved in: ./plots/")
    print("  - Traditional: pruning_results.png")
    print("  - Novel MAP: novel_map_pruning_results.png")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Check the plots in ./plots/ directory")
    print("2. Compare accuracy vs sparsity trade-offs")
    print("3. Analyze model file sizes in ./models/")
    print("4. Review training logs above for detailed metrics")
    print("="*80)

if __name__ == "__main__":
    main()
