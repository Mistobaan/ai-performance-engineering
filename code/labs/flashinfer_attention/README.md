# Lab - FlashInfer Block-Sparse Attention

## Summary
Runs a block-sparse attention kernel with FlashInfer and compares it to dense SDP + mask on an LLM-scale head configuration, including the output projection.

## Learning Goals
- Measure block-sparse attention speedups at high sparsity ratios.
- Validate FlashInfer kernels on realistic head dimensions.
- Profile attention + output projection as a unit of work.

## Files
| File | Description |
| --- | --- |
| `baseline_flashinfer_attention.py` | Dense SDP + mask baseline with output projection. |
| `optimized_flashinfer_attention.py` | FlashInfer block-sparse attention with output projection. |

## Running
```bash
python -m cli.aisp bench list-targets --chapter labs/flashinfer_attention
python -m cli.aisp bench run --targets labs/flashinfer_attention --profile minimal
```

## Notes
- Default head configuration targets gpt-oss-20b hidden size (2880) with head_dim=64 (45 heads).
- Increase `seq_len` if you need larger batch sizes to expose speedups.
- Requires FlashInfer (`pip install flashinfer-python==0.6.3`).

## Related Chapters
- **Ch16**: FlashInfer block-sparse attention.
