# Attribution and distribution notes

- Original Open Jev code: intikhab49/open-jev-typed-decision-engine, Apache-2.0. Source base commit 78d3b3a171f24d8d9a8dea18e027f9d3373fda45. This snapshot adds Core ML support and research/evaluation tooling; export packaging changes are recorded in docs/EXPORT_MANIFEST.json. Original copyright notices are retained in LICENSE and source files.
- ModernBERT-base: Answer.AI / LightOn / collaborators. Model architecture and weights are Apache-2.0 per https://huggingface.co/answerdotai/ModernBERT-base (checked 2026-10-06).
- Typed Decisions: LocalLLaMA/typed-decisions, dataset card declares Apache-2.0 at https://huggingface.co/datasets/LocalLLaMA/typed-decisions (checked 2026-10-06). Dataset files are not redistributed here.
- JudgeBench and private novllm corpus material were used for evaluation, not training these checkpoints. Dataset files, per-case predictions, review rows and raw API transcripts are omitted.

Model assets are distributed separately at https://huggingface.co/eightman999/open-jev-coreml . See MODEL_ASSETS.json there for hashes and experimental-model distinctions.
