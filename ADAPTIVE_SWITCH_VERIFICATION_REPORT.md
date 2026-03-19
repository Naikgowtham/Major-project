# Adaptive Exploration→Exploitation Switch Verification Report

**Date:** February 16, 2026  
**Implementation:** resnet20_adaptive.ipynb - MAPPruner Class  
**Status:** ✅ MOSTLY CORRECT with 2 MINOR ISSUES

---

## Executive Summary

Your adaptive switching implementation is **fundamentally sound** and implements all 5 required conditions. However, there are **2 bugs** that should be fixed:

1. **🐛 CRITICAL BUG**: Mask flip rate timing issue (off-by-one epoch)
2. **⚠️ MINOR ISSUE**: Loss plateau uses relative improvement instead of absolute difference

---

## Detailed Verification

### ✅ Condition 1: Sparsity Guard (S_t ≥ S_min)

**Status:** ✅ **CORRECT**

**Implementation:**
```python
current_sparsity = self.calculate_current_sparsity()

# Condition 1: Sparsity guard
if current_sparsity < self.min_sparsity:
    self.stable_epochs = 0  # Reset patience
    return False
```

**Verification:**
- ✅ Default S_min = 0.3 (30%)
- ✅ Correctly computes global sparsity across all layers
- ✅ Blocks switching when S_t < 0.3
- ✅ Resets patience counter when guard fails
- ✅ Uses correct comparison operator (<)

---

### 🐛 Condition 2: Mask Stability (Hamming Flip Rate)

**Status:** 🐛 **CRITICAL BUG - Timing Issue**

**Implementation:**
```python
# In should_switch_to_exploitation():
flip_rate = self.calculate_mask_flip_rate()
self.mask_flip_history.append(flip_rate)

if flip_rate < self.stability_threshold:
    self.stable_epochs += 1
else:
    self.stable_epochs = 0
```

```python
# In MAPConv2d.get_mask_flip_rate():
def get_mask_flip_rate(self):
    with torch.no_grad():
        flips = torch.sum(torch.abs(self.mask - self.prev_mask)).item()
        total = self.mask.numel()
        return flips / total if total > 0 else 0.0
```

**Verification:**
- ✅ Hamming distance formula is correct: Δ_t = |m_t - m_{t-1}|₀ / |m|
- ✅ Default ε = 0.005 (0.5%)
- ✅ Averages flip rate across all layers correctly
- ✅ Patience increments/resets correctly
- 🐛 **BUG**: Timing issue with when flip rate is calculated

**THE BUG:**

The flip rate is calculated at the **beginning** of `update_masks()` for epoch t, **before** masks are updated:

```
Epoch t-1 ends with: mask = M_{t-1}, prev_mask = M_{t-2}
Epoch t begins:
  1. should_switch_to_exploitation() calculates flip rate = |M_{t-1} - M_{t-2}| / |m|  ❌ WRONG
  2. Then update_mask() is called:
     - prev_mask = M_{t-1}
     - mask = M_t
```

**Problem:** At epoch t, you're comparing masks from epochs t-1 and t-2, not t and t-1!

**Impact:** Medium - The stability detection is delayed by one epoch. You're checking if masks from the *previous* epoch pair stabilized, not the current one.

---

### ✅ Condition 3: Patience (K Consecutive Epochs)

**Status:** ✅ **CORRECT**

**Implementation:**
```python
if flip_rate < self.stability_threshold:
    self.stable_epochs += 1
else:
    self.stable_epochs = 0  # Reset patience

if self.stable_epochs < self.patience:
    return False
```

**Verification:**
- ✅ Default K = 7
- ✅ Increments stable_epochs when Δ_t < ε
- ✅ Resets to 0 when stability breaks
- ✅ Requires K consecutive stable epochs
- ✅ Resets on sparsity guard failure

---

### ⚠️ Condition 4: Loss Plateau (L_{t-w} - L_t < δ)

**Status:** ⚠️ **MINOR ISSUE - Different Formula**

**Specification:**
```
L_{t-w} - L_t < δ  (absolute difference)
w = 10, δ = 0.005
```

**Your Implementation:**
```python
def check_loss_plateau(self):
    if len(self.loss_history) < self.loss_window:
        return False
    
    recent_losses = self.loss_history[-self.loss_window:]
    old_loss = recent_losses[0]   # L_{t-w}
    new_loss = recent_losses[-1]  # L_t
    
    # Check if improvement is negligible
    improvement = (old_loss - new_loss) / old_loss  # ❌ Relative, not absolute
    return improvement < self.loss_threshold
```

**Verification:**
- ✅ Default w = 10
- ✅ Default δ = 0.005 (0.5%)
- ✅ Waits for w epochs of history
- ✅ Compares oldest to newest in window
- ⚠️ **Uses relative improvement** `(L_{t-w} - L_t) / L_{t-w}` instead of **absolute difference** `L_{t-w} - L_t`

**Impact:** Low - Relative improvement is actually more robust for varying loss scales. The behavior is similar but technically different from specification.

**Mathematical Difference:**
- **Spec:** `L_{t-w} - L_t < 0.005` (absolute)
- **Your code:** `(L_{t-w} - L_t) / L_{t-w} < 0.005` (relative)

For a loss of ~2.0:
- Absolute: Improvement < 0.005 (plateau if loss drops < 0.005)
- Relative: Improvement < 0.5% (plateau if loss drops < 0.01)

---

### ✅ Condition 5: Permanent Mask Freezing

**Status:** ✅ **CORRECT**

**Implementation:**
```python
def update_masks(self, epoch, train_loss=None):
    # Check if we should switch to exploitation
    if not self.exploitation_started and self.should_switch_to_exploitation(epoch):
        self.exploitation_started = True
        self.switch_epoch = epoch
        # ... print switch message
    
    # If in exploitation mode, don't update masks
    if self.exploitation_started:
        current_sparsity = self.calculate_current_sparsity()
        self.sparsity_history.append(current_sparsity)
        return current_sparsity  # ✅ EARLY RETURN - no mask updates!
    
    # Otherwise, continue exploration with gradual pruning
    target_sparsity = self.get_target_sparsity_for_epoch(epoch)
    for layer in self.get_map_layers():
        layer.update_mask(target_sparsity)  # ← Never reached in exploitation
```

**Verification:**
- ✅ `exploitation_started` flag prevents re-triggering
- ✅ Early return skips all mask updates
- ✅ `switch_epoch` permanently recorded
- ✅ Sparsity remains constant (no new updates)
- ✅ Weights continue training, only masks frozen

---

### ✅ Training Loop Integration

**Status:** ✅ **CORRECT**

**Implementation:**
```python
def train_model(...):
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(...)
        test_loss, test_acc = test_epoch(...)
        scheduler.step()
        
        # Update masks with adaptive switching (pass train_loss for plateau detection)
        if pruner:
            current_sparsity = pruner.update_masks(epoch, train_loss)  # ✅ Once per epoch
            phase_info = pruner.get_phase_info()
```

**Verification:**
- ✅ `update_masks()` called **once per epoch** (not per batch)
- ✅ `train_loss` passed for plateau detection
- ✅ Check runs before mask updates
- ✅ No redundant calls
- ✅ Proper integration with training flow

---

## 🐛 Bugs and Issues Summary

### 1. 🔴 CRITICAL: Mask Flip Rate Timing Bug

**Location:** `MAPPruner.update_masks()` → `should_switch_to_exploitation()` → `calculate_mask_flip_rate()`

**Problem:** Flip rate is calculated before mask updates, comparing epoch (t-1, t-2) instead of (t, t-1).

**Fix Required:** Calculate flip rate AFTER masks are updated in the previous epoch.

### 2. 🟡 MINOR: Loss Plateau Formula Mismatch

**Location:** `MAPPruner.check_loss_plateau()`

**Problem:** Uses relative improvement `(L_{t-w} - L_t) / L_{t-w}` instead of absolute `L_{t-w} - L_t`.

**Fix Required:** Change to absolute difference (optional - relative may be better).

---

## 🔧 Exact Minimal Fixes Needed

### Fix #1: Correct Mask Flip Rate Timing

**Current flow (WRONG):**
```
Epoch t:
  1. calculate_mask_flip_rate()  ← compares M_{t-1} to M_{t-2}
  2. update_mask()               ← creates M_t
```

**Option A: Calculate flip rate AFTER previous epoch's updates**

Add flip rate calculation at the END of update_masks():

```python
def update_masks(self, epoch, train_loss=None):
    # Track loss
    if train_loss is not None:
        self.loss_history.append(train_loss)
    
    # If in exploitation mode, don't update masks
    if self.exploitation_started:
        current_sparsity = self.calculate_current_sparsity()
        self.sparsity_history.append(current_sparsity)
        return current_sparsity
    
    # Continue exploration with gradual pruning
    target_sparsity = self.get_target_sparsity_for_epoch(epoch)
    for layer in self.get_map_layers():
        layer.update_mask(target_sparsity)
    
    # Calculate flip rate AFTER masks are updated
    flip_rate = self.calculate_mask_flip_rate()
    self.mask_flip_history.append(flip_rate)
    
    # Check if we should switch to exploitation (check in NEXT epoch)
    if epoch >= 1:  # Need at least 2 epochs to compare
        if not self.exploitation_started and self.should_switch_to_exploitation(epoch):
            self.exploitation_started = True
            self.switch_epoch = epoch
            print(f"\n{'='*70}")
            print(f"🎯 ADAPTIVE SWITCH TO EXPLOITATION AT EPOCH {epoch}")
            print(f"{'='*70}")
            print(f"✓ Sparsity guard:     {self.calculate_current_sparsity():.3f} >= {self.min_sparsity}")
            print(f"✓ Mask stability:     {self.mask_flip_history[-1]:.4f} < {self.stability_threshold} (for {self.patience} epochs)")
            print(f"✓ Loss plateau:       Confirmed over {self.loss_window} epochs")
            print(f"🔒 MASKS FROZEN - Pure exploitation begins!")
            print(f"{'='*70}\n")
    
    current_sparsity = self.calculate_current_sparsity()
    self.sparsity_history.append(current_sparsity)
    return current_sparsity
```

**And modify should_switch_to_exploitation():**

```python
def should_switch_to_exploitation(self, epoch):
    """Check based on ALREADY CALCULATED flip rate from previous update"""
    if self.exploitation_started:
        return False
    
    if len(self.mask_flip_history) == 0:
        return False
    
    current_sparsity = self.calculate_current_sparsity()
    
    # Condition 1: Sparsity guard
    if current_sparsity < self.min_sparsity:
        self.stable_epochs = 0
        return False
    
    # Condition 2: Mask stability (use LAST calculated flip rate)
    flip_rate = self.mask_flip_history[-1]
    
    if flip_rate < self.stability_threshold:
        self.stable_epochs += 1
    else:
        self.stable_epochs = 0
    
    if self.stable_epochs < self.patience:
        return False
    
    # Condition 3: Loss plateau
    if not self.check_loss_plateau():
        return False
    
    return True
```

### Fix #2 (Optional): Use Absolute Loss Difference

```python
def check_loss_plateau(self):
    """Check if loss has plateaued over the window"""
    if len(self.loss_history) < self.loss_window:
        return False
    
    recent_losses = self.loss_history[-self.loss_window:]
    old_loss = recent_losses[0]
    new_loss = recent_losses[-1]
    
    # Use absolute difference as per specification
    improvement = old_loss - new_loss
    return improvement < self.loss_threshold
```

**Note:** You may want to keep relative improvement (it's more robust). If so, just document the deviation from spec.

---

## ✅ What Is Correct

1. ✅ All 5 conditions are implemented
2. ✅ Default parameters match specification
3. ✅ Sparsity guard works correctly
4. ✅ Patience mechanism works correctly
5. ✅ Masks freeze permanently after switch
6. ✅ Training loop integration is correct
7. ✅ Hamming distance formula is correct
8. ✅ Switch triggers only once
9. ✅ All tracking variables properly maintained
10. ✅ Clear diagnostic output when switching

---

## 🎯 Recommendation

**Priority:** Fix Bug #1 (mask flip rate timing) before running 300-epoch experiments.

**Optional:** Keep relative loss improvement (it's actually better than absolute for varying loss scales).

**Testing:** After fixes, verify with a short 20-epoch test that:
1. Flip rates stabilize correctly
2. Switch occurs at expected epoch
3. Masks remain frozen after switch
4. Sparsity stays constant in exploitation

---

## 📊 Overall Assessment

**Grade:** A- (90/100)

**Strengths:**
- Comprehensive implementation of all conditions
- Excellent code structure and clarity
- Proper state management
- Good diagnostic output

**Weaknesses:**
- Timing bug in flip rate calculation
- Minor formula deviation (relative vs absolute)

**Conclusion:** With the timing bug fixed, this implementation will be **robust and publication-ready**. The bug doesn't break the algorithm completely but delays switching by 1 epoch and may cause occasional false negatives in stability detection.
