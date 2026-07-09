# 合成データ生成手法ドキュメント＆開発・テストガイド

本ドキュメントは、本リポジトリで利用可能な合成データ生成手法（Generative Models）の現在の仕様、および現在も残っているハードコード要素、小規模検証用データセットを用いたテスト手順をまとめたものです。

---

## 1. 生成モデルの一覧と分類

本フレームワークに統合されている生成モデルは、**差分プライバシー（DP）対応モデル**と、**非プライベート（DP非対応）モデル**に大別されます。

### 差分プライバシー (DP) 対応モデル
個人のプライバシーを数学的に保証しながら合成データを生成するモデル群です。

*   **インターフェース共通仕様**: すべてのDP対応モデルにおいて、初期化時のデフォルト引数が **`epsilon=1.0`**, **`delta=1e-9`** に統一されています。
*   *(注: Pure $\epsilon$-DP である `PrivBayes` もインターフェース統一のため `delta` を受け取りますが、内部処理では無視されます。また、`TabDDPM` はデフォルトでDPが有効化されており、非プライベートで動作させたい場合は明示的に `epsilon=None` を指定します。)*

| モデル名 | 概要 | プライバシー定義 | ラッパー実装 | 元アルゴリズム / ソース |
| :--- | :--- | :--- | :--- | :--- |
| **PrivBayes** | ベイジアンネットワーク構築時に指数メカニズムと確率テーブルへのラプラスノイズ付加を用いる手法。 | **$\epsilon$-DP** (Pure) | [data_synthesiser.py](../generative_models/data_synthesiser.py) | DataSynthesizer |
| **AIM** | 適応的に低次元マージナル（周辺分布）を選択・計測し、Private-PGMでモデル化する手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [aim.py](../generative_models/aim.py) | AIM (PVLDB 2022) |
| **GEM** | 反復的DPに基づき、指定クエリ（ワークロード）に対する応答に適合するようにジェネレータ（WGAN）を学習させる手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [gem.py](../generative_models/gem.py) | GEM (NeurIPS 2021) |
| **DP_MERF** | ランダム特徴量と平均埋め込みによる最大エントロピー関係フォレストと、数値データのプライベート離散化（PrivTreeなど）を統合した手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [dp_merf.py](../generative_models/dp_merf.py) | DP-MERF (AISTATS 2021) |
| **PATE-GAN** | PATEフレームワークを利用し、複数の教師ディスクリミネータのノイズ付き投票を用いてGANをプライベートに学習する手法。 | **$(\epsilon, \delta)$-DP** (RDP経由) | [pate_gan.py](../generative_models/pate_gan.py) | PATE-GAN (ICLR 2019) |
| **Private-GSD** | プライベートに測定された周辺分布に適合するデータを、遺伝的アルゴリズム（GA）を用いて最適化・探索する手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [private_gsd.py](../generative_models/private_gsd.py) | Private-GSD (ICML 2023) |
| **PrivMRF** | プライベートな周辺分布の測定値から、マルコフ確率場（MRF）モデルを構築してサンプリングする手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [privmrf.py](../generative_models/privmrf.py) | PrivMRF (PVLDB 2021) |
| **PrivSyn** | 自動選択された1〜2次元マージナルに整合する合成データを段階的に再構成する手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [privsyn.py](../generative_models/privsyn.py) | PrivSyn (USENIX Security 2021) |
| **RAPpp** | 周辺分布クエリへの応答を連続空間上で勾配降下法により最適化した後、離散データへと射影・再構築する手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [rappp.py](../generative_models/rappp.py) | RAP++ (NeurIPS 2022) |
| **TabDDPM** | 表形式データに最適化されたデノイジング拡散確率モデル（DDPM）。Opacusを用いてDPを適用します。 | **$(\epsilon, \delta)$-DP** (RDP経由) | [tabddpm.py](../generative_models/tabddpm.py) | TabDDPM (ICML 2023) |

### 非プライベートモデル (DP非対応)
プライバシー保護（ノイズ付加等）を行わず、データの有用性や表現力を最大化するための標準的な生成モデル群です。

| モデル名 | 概要 | ラッパー実装 | 元アルゴリズム / ソース |
| :--- | :--- | :--- | :--- |
| **IndependentHistogram** | 属性間の依存関係を無視し、各列で独立してヒストグラムを計算・サンプリングするベースライン。 | [data_synthesiser.py](../generative_models/data_synthesiser.py) | DataSynthesizer |
| **BayesianNet** | GreedyBayesにより、ノイズを加えることなく属性間の依存関係（DAG）を学習してサンプリングする手法。 | [data_synthesiser.py](../generative_models/data_synthesiser.py) | DataSynthesizer |
| **CTGAN** | 表形式データの不均衡やマルチモーダルな数値分布に対応した条件付き生成敵対ネットワーク。 | [ctgan.py](../generative_models/ctgan.py) | CTGAN (NeurIPS 2020) |
| **GaussianMixtureModel** | scikit-learn の GMM（ガウス混合モデル）のラッパー。主に数値データ用。 | [gmm.py](../generative_models/gmm.py) | scikit-learn |

---

## 2. 各モデルに現在も残るハードコード要素・露出していないパラメータ

下層のアルゴリズム実装の仕様に依存する制限、または現時点でラッパー外部から変更できないパラメータのリストです。

### ① `DP_MERF`
*   **特徴量・エポック数の強制上書き**: 下層の [single_generator_priv_all.py](../method/DP_MERF/single_generator_priv_all.py) の `add_default_params` 関数内において、ラッパー側から指定された引数を無視して `n_features_arg = 2000` および `how_many_epochs_arg = 1000` に強制上書きされる実装が残っています。

### ② `PATE-GAN`
*   **ネットワーク構成の固定**: ジェネレータとディスクリミネータの隠れ層次元 `h_dim` が特徴量数と同値（`int(self.nfeatures)`）、ノイズの次元 `z_dim` が特徴量数の1/4（`int(self.nfeatures / 4)`）に固定されています。
*   **オプティマイザ設定**: Adamの `beta1=0.5` が固定されています。

### ③ `Private-GSD`
*   **計測クエリの固定**: 測定するマージナルの次数が `k=2`、ビンの分割数が `bins=[2, 4, 8, 16, 32]` に固定されています。
*   **GA設定**: 遺伝的アルゴリズムの集団サイズ `population_size_muta=50`、`population_size_cross=50`、早期終了フラグ `stop_early=True` が固定されています。
*   **乱数シード**: JAXのPRNGキーが `PRNGKey(0)` にハードコードされており、外部からの乱数シード設定がGAの初期状態に反映されません。

### ④ `RAPpp`
*   **最適化設定**: `RAPppConfiguration` において、学習率 `optimizer_learning_rate=[0.003]` 等のパラメータが固定されています。また、マージナルクエリの次数も `K=2` 固定です。（※かつて `iterations=[1]`, `sigmoid_doubles=[0]` と射影最適化が大幅に削減されていた欠陥は、本来の上流既定値である `iterations=[30]`, `sigmoid_doubles=[10]` へ修正されています）

### ⑤ `PrivSyn`
*   **データセット名**: ラッパー内部で `dataset='adult'` がハードコードされています。
*   **露出不足**: 数値データ前処理手法（`num_preprocess='privtree'`）、稀なカテゴリの閾値（`rare_threshold=0.005`）、一貫性保持ループ回数（`consist_iterations=501`）などがラッパーの引数として露出していません。

### ⑥ `GEM`
*   **ネットワーク次元**: ジェネレータの隠れ層の次元 `gen_dim=[dim * 2, dim * 2]` が固定されています。
*   **クエリ空間**: WGAN学習時に使用するマージナルの次数が `degree=2`、クエリ数 `workload=100000` に固定されています。

---

## 3. テスト用小規模データセットと実行ガイド

開発時の動作検証を高速に行うため、ルートの `tasks` ディレクトリに小規模なデータセットと設定ファイルを用意しています。

### ① テスト用構成要素
*   **データセット**: 
    *   データ: `tasks/data/texas_mini.csv` (元の texas データセットからヘッダー+1000レコードを抽出)
    *   スキーマ定義: `tasks/data/texas_mini.json`
*   **設定ファイル**: `tasks/runconfig.json` (反復回数1回、エポック数やタイムステップ数を極小にしたテスト用軽量設定)
*   **出力先**: `tasks/outputs/` 以下に結果のJSONやキャッシュが生成されます。

### ② 検証テストの実行コマンド
UTF-8 環境変数およびクラウド同期制限対策を指定した状態で、以下のコマンドを実行します。

```powershell
# PowerShell での実行例 (Windows環境)
$env:PYTHONUTF8=1; $env:UV_LINK_MODE="copy"; uv run python utility_cli.py -D tasks/data/texas_mini -RC tasks/runconfig.json -O tasks/outputs/utility --device cpu
```

このコマンドを実行することで、`PrivBayes`, `DP_MERF`, `TabDDPM` の3つの異なるDPモデルが軽量な設定で順次トレーニング・サンプル生成される様子を数分以内で動作確認することができます。

---

## 4. εのみが与えられた場合の動作（δの取り扱い）

ユーザーが $\epsilon$ のみを与えて（すなわち $\delta$ の指定を省略、あるいは純粋な $\epsilon$-DP を期待して $\delta=0$ を明示的に設定して）実行した場合、モデルによって挙動が異なります。

*   **`PrivBayes` (Pure $\epsilon$-DP)**:
    *   元々 $\epsilon$-DP のアルゴリズムであるため、$\epsilon$ のみで設計通り完全に動作します（$\delta$ は不要です）。
*   **zCDP 依存モデル (`AIM`, `GEM`, `Private-GSD`, `PrivMRF`, `PrivSyn`, `RAPpp`)**:
    *   プライバシー予算の変換処理 `cdp_rho` 内で `assert delta > 0` を宣言しているため、$\delta=0$ を明示的に指定すると **`AssertionError` でプログラムが異常終了します**。
    *   $\delta$ を省略した場合（未指定時）は、デフォルト値 `delta=1e-9` が自動適用され、**$(\epsilon, 10^{-9})$-DP として動作**します。
*   **`PATE-GAN`**:
    *   PATE のノイズスケール計算式において $\delta > 0$ であることが前提となっており、$\delta = 0$ では正しく動作しません。省略時はデフォルトの `delta=1e-9` が使用されます。
*   **`DP_MERF`**:
    *   ラッパーの引数は `epsilon` のみですが、内部ではその値をそのまま zCDP の $\rho$（`rho = epsilon`）として扱っています。指定した $\epsilon$ よりも緩いプライバシー保証となるため、純粋な $\epsilon$-DP 相当の強度では動作しません。
