# Export validation — 2026-10-06

- Source snapshot: see EXPORT_MANIFEST.json. Original working tree was clean and was not edited.
- Original tracked files: 100. Selected: 94. Excluded: five JSONL files and reference_predictions.json. Untracked source artifacts/datasets were not copied.
- No JSONL, notebook output cells, model binaries or files larger than 10MB in GitHub export. Common token/private-key pattern scan found no matches. This is a bounded scan, not a proof that arbitrary text contains no sensitive information.
- Existing portable pytest selection over worker/math/serialization/probe/JudgeBench tests: 47 PASS, 17 deselected.
- Full serialization/probe/JudgeBench modules: 39 PASS. These overlap the previous selection; do not add the counts as unique tests.
- Checkpoints inspected with torch.load(weights_only=True, map_location="cpu"). Only state_dict and model/probe metadata keys present; no dataset fields.
- HF asset manifest: 35 files, 8,056,735,510 bytes excluding model card/license/asset index. Model/checkpoint bytes unchanged. Package manifests declare historical slots=none and Hub-relative paths.
- No retraining, dataset download, hosted inference or new Core ML/ANE prediction performed. Historical benchmark results retain their original scope.
- Explicit pure-math parity selection: 2 PASS, 10 deselected.
- GitHub portable CI passed on initial public commit 71dfa0a2aee347cec09771c7122382ed21dbdfdc: https://github.com/eightman999/open-jev-coreml/actions/runs/37436254385 .
