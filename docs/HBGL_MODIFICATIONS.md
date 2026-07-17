# HBGL modification record

## Scope and upstream base

The local source was added in repository commit `62ad1a3` on 2024-02-05 and was
compared file-by-file with `kongds/HBGL` commit
`a40acdf87407a5a6cdd4c921c80c60b9f3522aa1`. The copied implementation retains
substantial HBGL and Microsoft `s2s-ft` source. No top-level licence was found in
the inspected upstream repository.

The apparent full-file changes under `s2s_ft` are largely CRLF/LF conversion.
Ignoring line endings, `config.py` is identical and the functional differences
in `convert_state_dict.py`, `modeling.py`, `modeling_decoding.py` and
`s2s_loader.py` are small. `utils.py`, the refactored `main/` training package,
`test.py`, `eval.py` and `preprocess.py` contain the substantive study changes.
The inspected commit is therefore a defensible comparison base, although the
students' original fork history was not preserved.

## Study-specific changes

| Area | Result-affecting or operational changes |
| --- | --- |
| Training structure | Splits upstream's monolithic `run.py` into `main/`; adds offline W&B logging, separate tester helpers, job IDs and newer Transformers compatibility. |
| Label initialisation | Adds random initialisation, NYT leaf-name initialisation, expanded RCV1 descriptions, self-attention controls and split conceptual pre-training for hierarchies too large to process together. |
| Data loading | Truncates soft-label target depth and replaces serial feature creation with multiprocessing and cached features. |
| Decoding | Adds large-file loading, meta-label filtering, result files, P@1/2/3/5/10/25/50, R-Precision and study seed/job metadata. |
| Cross-domain runs | Adds AmazonCat-13K, Wiki10-31K and an unsuccessful Amazon-670K configuration alongside WOS, NYT and RCV1. |

## Preserved configurations

All working launchers use BERT base uncased, learning rate `3e-5`, 96,000
training steps, warm-up 500, and seeds `42, 1, 2, 3, 4`.

| Dataset | Source / target length | Batch | Hierarchy-specific options |
| --- | ---: | ---: | --- |
| WOS | 509 / 3 | 12 | conceptual pre-training, BCE, 300 steps |
| NYT | 472 / 9 | 12 | leaf-name initialisation, conceptual pre-training, BCE, 1,000 steps at `1e-4` |
| RCV1 | 492 / 5 | 12 | expanded label descriptions, random initialisation, conceptual pre-training, BCE, 100 steps |
| AmazonCat-13K | 500 / 4 | 12 | random initialisation, ignore synthetic meta-labels |
| Wiki10-31K | 500 / 4 | 12 | random initialisation, ignore synthetic meta-labels |
| Amazon-670K | 500 / 5 | 16 | random initialisation; script is labelled OOM and unverified |

## Release-readiness fixes

- Invalid generated tokens are now excluded instead of silently becoming label
  zero; malformed or empty gold labels fail evaluation.
- P@k handles short prediction vectors, R-Precision rejects empty-gold examples,
  and metric helpers have focused unit tests.
- `--no_cuda` is honoured and a hard-coded CUDA allocation in decoding now uses
  the prediction tensor's device.
- Restored the missing TensorBoard import, made CPU batch-size calculation
  non-zero, bounded the debug sample loop and removed an extra training step.
- Test evaluation now receives the training seed and job ID and avoids decoding
  the same checkpoint twice when macro- and micro-best paths coincide.
- Conceptual-pre-training embeddings are scoped to the seed output directory;
  cross-dataset reuse of a global `trained_label_embeddings.pt` is prevented.
- The five-seed launchers keep separate seed directories and refuse to overwrite
  existing runs. Earlier scripts recursively deleted the preceding seed output.
- Three derived NYT JSON files containing text were removed and are now ignored;
  they remain recoverable from Git history.

## Known limitations

- No training or inference path has been run in a fresh environment during this
  review. The environment pins were preserved from student work, not established
  by a lock file or successful clean installation.
- The `main/` refactor and split conceptual pre-training are extensive and lack
  automated model-level tests. The Amazon-670K launcher is explicitly known to
  run out of memory.
- Multiprocessing passes a Transformers tokenizer to worker processes and holds
  both input records and tokenised results in memory. Portability and peak memory
  use have not been measured.
- Multi-GPU behaviour has not been revalidated after the refactor.
- P@k uses a denominator of `k` even when decoding returns fewer than `k` labels.
  This is standard XMC precision-at-k behaviour but must match the paper tables.
- The retained taxonomy and label-vocabulary files may be derived from source
  datasets. Their redistribution status still needs explicit review.
- Upstream HBGL and the copied `s2s-ft` subset do not include a clear top-level
  licence in the inspected checkout. Permission or licence clarification is a
  publication blocker for redistribution, even though attribution is recorded.
