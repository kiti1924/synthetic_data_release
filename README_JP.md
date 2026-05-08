# 合成データ公開のためのプライバシー評価フレームワーク
合成データ公開におけるプライバシーと有用性のトレードオフを評価するための実用的なフレームワークです。

このフレームワークは、Stadler, T., Oprisanu, B., Troncoso, C. による論文「In 31st USENIX Security Symposium (USENIX Security22), pages 1451–1468, Boston, MA. USENIX Association.」（2022年）に基づいています。
- 公式論文: https://www.usenix.org/conference/usenixsecurity22/presentation/stadler
- arXiv: https://arxiv.org/abs/2011.07018
- GitHub: https://github.com/spring-epfl/synthetic_data_release

## 更新履歴 (Updates)
本フォークにおける主要な更新要素（MPI分散処理への対応、および堅牢な実行キャッシュ機構の導入など）の詳細については、[UPDATE_JP.md](UPDATE_JP.md) を参照してください。

# 攻撃モデル
`attack_models` モジュールには、現在次の攻撃モデルが含まれています。

- リンケージ攻撃をメンバーシップ推論攻撃としてモデル化したプライバシー敵対者 `MIAAttackClassifier` によるプライバシーゲイン評価
- ターゲットレコードの一部の情報を知っている場合に、ターゲットの機微な値を推測することを目的とした単純な属性推論攻撃 `AttributeInferenceAttack`

# 生成モデル
`generative_models` モジュールには、現在次のモデルが含まれています。

- `IndependentHistogram`: [Data Responsibly の DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer) から適応した独立ヒストグラムモデル
- `BayesianNet`: [Data Responsibly の DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer) から適応したベイジアンネットワークに基づく生成モデル
- `PrivBayes`: [Data Responsibly の DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer) から適応したベイジアンネットワークモデルの差分プライバシー版
- `CTGAN`: [CTGAN](https://github.com/sdv-dev/CTGAN) の CTGAN モデルを統合した条件付き表形式生成敵対ネットワーク
- `PATE-GAN`: [MLforHealth Lab](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/82d7f91d46db54d256ff4fc920d513499ddd2ab8/alg/pategan/) の元実装を基に適応された差分プライバシー生成敵対ネットワーク
- その他のモデルとして `AIM`, `GEM`, `PrivMRF`, `PrivSyn`, `DP_MERF`, `TabDDPM`, `PrivateGSD`, `RAPpp` などを含みます。

# セットアップ

## 直接インストール

### 必要条件
このフレームワークとその構成要素は Python 3.9+ で開発およびテストされています。
このプロジェクトは依存関係の管理に [uv](https://docs.astral.sh/uv/) を使用しています。すべての依存関係（CTGAN フォークを含む）は `pyproject.toml` に宣言されています。

**標準的な（CPUのみの）環境向け:**
```bash
uv sync
```

**GPUで加速する環境向け（CUDA 12）:**
対応する NVIDIA GPU をお持ちの場合、オプションの GPU 依存関係（NVIDIAのパッケージレジストリから直に `cuml-cu12`, `cudf-cu12`, `cupy-cuda12x` を取得します）をインストールすることで、処理速度を飛躍的に向上させることができます：
```bash
uv sync --extra gpu
```

インストールが正しく行われたか確認するには、次のコマンドを実行してください。
```bash
uv run python -c "import ctgan"
uv run python -c "import cuml; print('GPU support ready!')"  # --extra gpu を利用した場合のみ
```


# 実行例

このリポジトリでは、Utility、Linkage（MIA）、Inference の評価を共通実行エンジン `all_cli.py` でまとめて実行できます。これにより、共通データセットを何度も読み込む必要がなくなります。個別の CLI でそれぞれの評価を実行することもできます。

### 統合実行（推奨）

3つの評価をシームレスに実行できます。CLI は適切なワーカーを自動スケジューリングし、PyTorch/CUDA リソースを動的に管理します。

```bash
uv run python all_cli.py -D data/texas -O outputs/texas -W 4 --device cpu
```

このコマンドは `-RCU`（Utility）、`-RCL`（Linkage）、`-RCI`（Inference）の設定を順次実行します。これらはデフォルトで `tests/*/runconfig.json` テンプレートを使用します。

### 個別評価の実行

```bash
# リンケージプライバシー評価（MIA ベース）
uv run python linkage_cli.py -D data/texas -RC tests/linkage/runconfig.json -O tests/linkage --device cpu

# 推論プライバシー評価（属性推論）
uv run python inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference --device cuda:0

# ユーティリティ評価
uv run python utility_cli.py -D data/texas -RC tests/utility/runconfig.json -O tests/utility
```

### デバイス最適化とハードウェアについて（GPU/CPU）

ほとんどのモデル（`AIM`, `GEM`, `TabDDPM`, `DP_MERF`, `CTGAN`, `PATEGAN`, `PrivMRF` など）は、CPU/GPU 両方の並列実行に対応するよう検証および修正されています。

1. **`--device` フラグ**: `cpu`, `cuda:0`, `cuda:1` などをサポートします。指定がない場合、CUDA が利用可能なら動的に選択します。
2. **環境変数**: `export SYNTHETIC_DATA_DEVICE="cuda:0"` で実行コンテキストを全体設定できます。
3. **スマート GPU 割り当て**: `SYNTHETIC_DATA_DEVICE` が `gpu` または `cuda` を示し、かつ現在のアルゴリズムが `GPU_MODELS` リストに含まれる場合、フレームワークは CUDA コンテキストの競合と VRAM OOM を避けるため、`multiprocessing` を `max_workers=1` に制限します。
4. **共有 IPC**: このフレームワークは `joblib.Parallel (loky)` を使用し、重い入力データセットを OS 共有メモリ経由でワーカーに送信することで、ギガバイト単位の重複ピクル化を回避します。
5. **Sklearn パイプライン**: 内部データ表現は `sklearn.compose.ColumnTransformer` をネイティブに利用しており、属性の One-Hot エンコーディングや `StandardScaler` の適用を Python ループのオーバーヘッドなしに処理します。

## 結果の解析

出力される JSON ファイルは、`utils/analyse_results.py` 内のヘルパー関数 `load_results_linkage`、`load_results_inference`、`load_results_utility` を使ってノートブック上で解析できます。ROC 曲線、ユーティリティ、およびアドバンテージ差分をプロットするのに便利です。

## ライセンスとサードパーティ製ソフトウェア

本リポジトリはオリジナルのコードに加え、複数のサードパーティ製アルゴリズムを含んでいます。
詳細については、ルートディレクトリにある各 `LICENSE.*` ファイル、および `method/` 以下の各ディレクトリ内の `LICENSE` ファイルを参照してください。

| コンポーネント / ディレクトリ | 適用ライセンス | ルートのライセンスファイル |
| :--- | :--- | :--- |
| **メインリポジトリ** (オリジナルコード) | BSD-3-Clause | `LICENSE` |
| `method/RAP/` | CC BY-NC 4.0 | `LICENSE. CC_BY-NC_4.0` |
| `method/AIM/` | Apache License 2.0 | `LICENSE.APACHEV2` |
| `method/PrivMRF/` | Apache License 2.0 | `LICENSE.APACHEV2` |
| `CTGAN` (フォークした依存関係) | MIT License | `LICENSE.MIT` |
| `k_anonymization` (フォークした依存関係) | Clear BSD License | `LICENSE.CLEAR_BSD` |
| `method/DP_MERF/` | MIT License | `LICENSE.MIT` |
| `method/TabDDPM/` | MIT License | `LICENSE.MIT` |
| `DataSynthesizer` (適応利用) | MIT License | `LICENSE.MIT` |

> [!WARNING]
> ### 非営利目的の制限
> `method/RAP/` コンポーネントは厳密に **非営利目的** に限定されており、**クリエイティブ・コモンズ 表示-非営利 4.0 国際 (CC BY-NC 4.0)** の下でライセンスされています。
> このリポジトリを商用利用する場合、コンプライアンスを維持するために `method/RAP/` ディレクトリを削除する必要がある場合があります。

## 謝辞
コードの一部は以下のリポジトリを利用・適応しています： [DP Tabular Data Synthesis Benchmark](https://github.com/KaiChen9909/tab_bench), [AIM](https://github.com/ryan112358/private-pgm), [DP-MERF](https://github.com/ParkLabML/DP-MERF), [GEM](https://github.com/terranceliu/iterative-dp?tab=readme-ov-file), [Private-GSD](https://github.com/giusevtr/private_gsd), [PrivMRF](https://github.com/caicre/PrivMRF), [PrivSyn](https://github.com/agl-c/deid2_dpsyn), [RAP++](https://github.com/amazon-science/relaxed-adaptive-projection), [TabDDPM](https://github.com/yandex-research/tab-ddpm)。コミュニティへの貢献に心より感謝申し上げます。
