# HGCLR integration

This directory contains the integration of HGCLR used in the cross-domain
HTC/XMC study. HGCLR was introduced by Wang et al. (ACL 2022); it is not a model
developed by this project.

- Original repository: <https://github.com/wzh9969/contrastive-htc>
- Upstream base commit: `322a7ff2d83c878534bed25bb288cf4479d00363`
- Upstream licence: MIT - retained in [LICENSE](LICENSE)
- Original documentation: [UPSTREAM_README.md](UPSTREAM_README.md)
- Study usage: [USAGE.md](USAGE.md)

## Study-specific changes

- preprocessing from the cleaned HTC JSONL representations used in the study;
- P@1, P@3, P@5 and R-Precision evaluation;
- training and inference cost instrumentation;
- compatibility with the modern H100/PyTorch environment used for the final runs;
- sequential and bwUniCluster five-seed launch scripts;
- deterministic aggregation of per-seed metrics.

The imported implementation also includes small release-hardening changes:

- machine-specific input paths were replaced with required command-line paths;
- preprocessing rejects unknown or empty-gold labels instead of dropping them;
- R-Precision rejects empty-gold samples explicitly;
- CUDA is synchronised at cost-timing boundaries;
- the legacy NumPy compatibility shim no longer uses `eval()`.


## Generated files

Model checkpoints and generated `.bin`, `.idx` and `.pt` dataset files are not
stored in Git. They are recreated through the documented preprocessing and
training commands.

## Citation

```bibtex
@inproceedings{wang-etal-2022-incorporating,
  title = {Incorporating Hierarchy into Text Encoder: a Contrastive Learning Approach for Hierarchical Text Classification},
  author = {Wang, Zihan and Wang, Peiyi and Huang, Lianzhe and Sun, Xin and Wang, Houfeng},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year = {2022},
  url = {https://aclanthology.org/2022.acl-long.491/}
}
```
