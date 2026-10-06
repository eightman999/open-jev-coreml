# Open Jev Core ML / ANE

Open Jev / JevLiteのApple Silicon向けCore ML実装と実験記録です。encoderとtoken score headをCore MLへ変換し、動的な質問・ラベル処理はCPU側で行います。

上流: [intikhab49/open-jev-typed-decision-engine](https://github.com/intikhab49/open-jev-typed-decision-engine)。コードはApache-2.0。本リポジトリは2026-10-06に既存フォークから抽出しました。**モデル重みは[Hugging Face](https://huggingface.co/eightman999/open-jev-coreml)で配布し、ここにはコード・実験記録を収録しています。データセットは配布しません。**

## 過去実測と限界

2026-09-21〜22、M2 Max 32GBでの記録。抽出時の再実験結果ではありません。

- 元checkpointのtyped-decisions test accuracy: **0.6430**。上流ensembleの0.6965とは異なります。
- Core ML fp16とTorchの判定一致率: **99.65%**（400件・2,000判定）。
- cpu_and_neのmedian latency: **13.2 / 37.1 / 111.1 ms**（256 / 512 / 1024 tokens）。長い入力ではTorch MPSが速い結果でした。
- 演算ごとのANE配置・消費電力は未測定。速度差だけから全演算のANE実行を断定しません。
- corpus cleaningで歴史的Jev判定との一致率は0.15〜0.24%。正解率ではありません。
- JudgeBench accuracy（保留を誤答として数える）は0.112、保留率0.761。汎用Jev代替としての実用性は確認できていません。
- 入力形式の位置バイアス修正・診断再学習後も用途外の精度改善は未達です。

## 実行

過去実測はPython 3.12。依存指定は範囲指定で、完全固定lockfileではありません。環境はartifacts/environment.jsonを参照してください。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install huggingface_hub
hf download eightman999/open-jev-coreml jevlite.pt --local-dir .
hf download eightman999/open-jev-coreml --include 'coreml/**' --local-dir artifacts
python 05_serve.py --backend coreml --compute-units cpu_and_ne --demo
```

Core ML推論は子プロセス隔離が既定。inprocessは診断用です。macOS15 targetのfp16で精度崩壊を観測したためmacOS26 targetを使います。既存の古いcheckpointと最新の入力形式の互換性は、各レポートとcheckpointのslots設定を確認してください。

学習・評価には外部データと基底モデルが別途必要です。学習スクリプトは実行すると外部取得を行い得ます。外部API評価は認証・利用料金が別途必要です。配布版のCore ML manifestは古い主checkpointに合わせslots=noneを明示します。Torch側で主checkpointを使う場合はDecisionEngine(..., slots="none")を指定します。診断用モデルはHFのexperimental/で別配布しています。

## 記録

- [ANE_REPORT.md](ANE_REPORT.md)
- [BASE_MODEL_POSITION_AUDIT.md](BASE_MODEL_POSITION_AUDIT.md)
- [BENCHMARK.md](BENCHMARK.md)
- [DIAG_RETRAIN.md](DIAG_RETRAIN.md)
- [JEVLITE_NEVER_A_AUDIT.md](JEVLITE_NEVER_A_AUDIT.md)
- [JEV_COMPETITION.md](JEV_COMPETITION.md)
- [JEV_LLM_SILVER_JUDGE.md](JEV_LLM_SILVER_JUDGE.md)
- [JUDGEBENCH_EXTERNAL_GOLD.md](JUDGEBENCH_EXTERNAL_GOLD.md)
- [SERIALIZATION_FIX.md](SERIALIZATION_FIX.md)

- [元フォーク変更履歴](docs/SOURCE_HISTORY.txt)
- [抽出元・加工記録・ファイルhash](docs/EXPORT_MANIFEST.json)
- [未同梱モデルのhash・サイズ](docs/MODEL_ASSETS.json)
- [抽出時検証結果](docs/VALIDATION.md)
- [上流README保存版](docs/UPSTREAM_README.md) — 上流の性能をこのforkの性能と混同しないでください。

初期監査の原因解釈は後続調査で更新されています。BASE_MODEL_POSITION_AUDIT、SERIALIZATION_FIX、DIAG_RETRAINも併読してください。

## 配布範囲

コード、テスト、技術レポート、集計JSONを収録。データセット本体、ケース別入力・予測、レビューサンプル、API生ログ、認証情報、会話全文、モデルバイナリは含めません。元Git履歴にもケース別データがあるため継承していません。

レポート中の省略対象ファイルへの参照は過去実験の出典名として残しています。データがないため完全な評価再現はこのrepoだけではできません。モデル重みの配布元とライセンス確認は[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)を参照してください。

## Portable checks

```bash
python -m pytest tests/test_worker.py -k 'not real' -q
python -m pytest tests/test_coreml_parity.py -k 'linear_before_after or grouped_softmax' -q
python -m pytest tests/test_serialization.py tests/test_posprior_probe.py tests/test_judgebench.py -q
```

これらはCore ML実機推論やモデル精度の検証とは別です。
