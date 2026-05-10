# Read-Process-Write Sorting Reproduction

This repository reproduces the Read-Process-Write architecture from the paper *Order Matters: Sequence to Sequence for Sets* using PyTorch.

## Current Progress

### Implemented

- [x] Synthetic sorting dataset
- [x] Read module: maps each scalar input into a memory embedding using an MLP
- [x] Process module: performs multiple attention-based processing steps over input memories
- [x] Write module: uses a Pointer Network-style decoder to output sorted input indices
- [x] Training script for the basic sorting task
- [x] Evaluation script using exact-match sorting accuracy
- [x] Masking mechanism to prevent repeated selection

### Not Yet Implemented

- [ ] Glimpse attention before pointer output
- [ ] Baseline Pointer Network model
- [ ] Experiments for different set sizes, such as N = 5, 10, 15
- [ ] Comparison between RPW model and baseline Ptr-Net
- [ ] Experiment table similar to the paper

## Next Steps

The next stage is to extend the current Write module with glimpse attention.  
After that, we will implement the baseline Pointer Network and compare it with the Read-Process-Write model on the sorting task.