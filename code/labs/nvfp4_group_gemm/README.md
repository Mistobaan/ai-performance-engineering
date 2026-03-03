# NVFP4 Group GEMM

This is the single canonical NVFP4 grouped GEMM lab.

## Run

List targets:
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm
```

Run baseline vs optimized:
```bash
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile deep_dive --update-expectations
```

## Layout

- `baseline_nvfp4_group_gemm_case{0,1,2,3}.py`: baseline wrappers
- `optimized_nvfp4_group_gemm_case{0,1,2,3}.py`: final optimized wrappers
- `custom_cuda_submission.py`: runtime + extension integration (`prepare_custom_cuda`, `custom_kernel_custom_cuda`)
- `custom_cuda_group_gemm_kernel.cu`: CUDA kernel implementation
- `expectations_b200.json`: current benchmark expectation snapshot
- `WORKLOG.md`: concise findings and final configuration notes
