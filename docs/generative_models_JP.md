# 合成データ生成手法ドキュメント

本リポジトリで利用可能な合成データ生成手法（Generative Models）の概要、および各モデルにおけるハイパーパラメータの実装状況（ハードコードや上書き等の問題を含む）に関するドキュメントです。

---

## 1. 生成モデルの一覧と分類

本フレームワークに統合されている生成モデルは、**差分プライバシー（DP）対応モデル**と、**非プライベート（DP非対応）モデル**に大別されます。

### 差分プライバシー (DP) 対応モデル
個人のプライバシーを数学的に保証しながら合成データを生成するモデル群です。内部で **$\epsilon$-DP (Pure DP)** を満たす手法と、**$(\epsilon, \delta)$-DP (Approximate DP)** を満たす手法（内部的には zCDP や RDP を使用）に分かれます。

| モデル名 | 概要 | プライバシー定義 | ラッパー実装 | 元アルゴリズム / ソース |
| :--- | :--- | :--- | :--- | :--- |
| **PrivBayes** | ベイジアンネットワーク構築時に指数メカニズムと確率テーブルへのラプラスノイズ付加を用いる手法。 | **$\epsilon$-DP** (Pure) | [data_synthesiser.py](../generative_models/data_synthesiser.py) | DataSynthesizer |
| **AIM** | 適応的に低次元マージナル（周辺分布）を選択・計測し、Private-PGMでモデル化する手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [aim.py](../generative_models/aim.py) | AIM (PVLDB 2022) |
| **GEM** | 反復的DPに基づき、指定クエリ（ワークロード）に対する応答に適合するようにジェネレータ（WGAN）を学習させる手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [gem.py](../generative_models/gem.py) | GEM (NeurIPS 2021) |
| **DP_MERF** | ランダム特徴量と平均埋め込みによる最大エントロピー関係フォレストと、数値データのプライベート離散化（PrivTreeなど）を統合した手法。 | **$\epsilon$-DP** (注1) | [dp_merf.py](../generative_models/dp_merf.py) | DP-MERF (AISTATS 2021) |
| **PATE-GAN** | PATEフレームワークを利用し、複数の教師ディスクリミネータのノイズ付き投票を用いてGANをプライベートに学習する手法。 | **$(\epsilon, \delta)$-DP** (RDP経由) | [pate_gan.py](../generative_models/pate_gan.py) | PATE-GAN (ICLR 2019) |
| **Private-GSD** | プライベートに測定された周辺分布に適合するデータを、遺伝的アルゴリズム（GA）を用いて最適化・探索する手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [private_gsd.py](../generative_models/private_gsd.py) | Private-GSD (ICML 2023) |
| **PrivMRF** | プライベートな周辺分布の測定値から、マルコフ確率場（MRF）モデルを構築してサンプリングする手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [privmrf.py](../generative_models/privmrf.py) | PrivMRF (PVLDB 2021) |
| **PrivSyn** | 自動選択された1〜2次元マージナルに整合する合成データを段階的に再構成する手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [privsyn.py](../generative_models/privsyn.py) | PrivSyn (USENIX Security 2021) |
| **RAPpp** | 周辺分布クエリへの応答を連続空間上で勾配降下法により最適化した後、離散データへと射影・再構築する手法。 | **$(\epsilon, \delta)$-DP** (zCDP経由) | [rappp.py](../generative_models/rappp.py) | RAP++ (NeurIPS 2022) |

*(注1: `DP_MERF` は理論上 $(\epsilon, \delta)$-DP や zCDP に対応していますが、本リポジトリのラッパー上は `epsilon` のみを受け取る形となっており、さらにその値をそのまま zCDP の $\rho$ として下層に流す実装上のバグ/考慮不足が存在します。詳細は第2節を参照。)*

### 非プライベートモデル (DP非対応)
プライバシー保護（ノイズ付加等）を行わず、データの有用性や表現力を最大化するための標準的な生成モデル群です。

| モデル名 | 概要 | ラッパー実装 | 元アルゴリズム / ソース |
| :--- | :--- | :--- | :--- |
| **IndependentHistogram** | 属性間の依存関係を無視し、各列で独立してヒストグラムを計算・サンプリングするベースライン。 | [data_synthesiser.py](../generative_models/data_synthesiser.py) | DataSynthesizer |
| **BayesianNet** | GreedyBayesにより、ノイズを加えることなく属性間の依存関係（DAG）を学習してサンプリングする手法。 | [data_synthesiser.py](../generative_models/data_synthesiser.py) | DataSynthesizer |
| **CTGAN** | 表形式データの不均衡やマルチモーダルな数値分布に対応した条件付き生成敵対ネットワーク。 | [ctgan.py](../generative_models/ctgan.py) | CTGAN (NeurIPS 2020) |
| **TabDDPM** | 表形式データ（数値/カテゴリ混在）に最適化されたデノイジング拡散確率モデル（DDPM）。 | [tabddpm.py](../generative_models/tabddpm.py) | TabDDPM (ICML 2023) |
| **GaussianMixtureModel** | scikit-learn の GMM（ガウス混合モデル）のラッパー。主に数値データ用。 | [gmm.py](../generative_models/gmm.py) | scikit-learn |

---

## 2. パラメータ統一化とバグ修正の実施（適用済み）

本フレームワークの拡張性と一貫性を高めるため、以下の修正・パラメータ統一化を実施しました。

### ① `epsilon` および `delta` のデフォルト引数の統一
*   **内容**: すべてのDP対応モデルラッパー（`AIM`, `GEM`, `DP_MERF`, `PATE-GAN`, `Private-GSD`, `PrivMRF`, `PrivSyn`, `RAPpp`, `TabDDPM`, `PrivBayes`）について、初期化時のデフォルト引数を `epsilon=1.0`, `delta=1e-5` に統一しました。
*   **注記**: Pure $\epsilon$-DP である `PrivBayes` は、インターフェース統一のために `delta` を受け取りますが、内部処理では無視されます。また、`TabDDPM` はデフォルトでDPが有効化されており、非プライベートで動作させたい場合は明示的に `epsilon=None` を指定します。

### ② `TabDDPM` のDP有効化（修正完了）
*   **修正内容**: ラッパー側の `tabddpm.py` において、これまで `None` で固定されていた `dp_epsilon`, `dp_delta`, `rho_used` に対し、指定された `epsilon` と `delta` から `cdp_rho` を用いて計算した値を渡すように修正しました。
*   **効果**: ラッパー経由でも差分プライベートな表形式拡散モデル（Opacusの `PrivacyEngine` を用いた学習）が正常に機能するようになりました。

### ③ `DP_MERF` のパラメータ型誤変換と露出の修正（修正完了）
*   **修正内容**: 
    1. `__init__` に `delta=1e-5` を追加し、他モデルと統一。
    2. `fit()` 内で `cdp_rho(self.epsilon, self.delta)` をインポートして zCDP の $\rho$ を算出し、前処理およびモデル学習の `merf_main` に渡すように変更しました（従来は生の $\epsilon$ を $\rho$ としてそのまま渡していたバグを解消）。
*   *(注: 下層の `single_generator_priv_all.py` 内に、指定された `n_features_arg` や `how_many_epochs_arg` をデフォルト値で強制上書きしてしまう実装が含まれていますが、ラッパーから渡された $\rho$ 値によるDPノイズの加算処理は正しく機能するようになりました。)*

### ④ `PATE-GAN` の引数名統一（修正完了）
*   **修正内容**: `__init__` の引数名を `eps` から `epsilon` に変更し、デフォルト値を `epsilon=1.0`, `delta=1e-5` に統一しました。後方互換性のため、`eps` もキーワード引数として受け取り可能にしています。

### ⑤ `PrivBayes` のデフォルト値および引数統一（修正完了）
*   **修正内容**: デフォルトの `epsilon` を `0.1` から `1.0` に変更し、インターフェース統一のために `delta=1e-5` を追加（内部では使用せず無視）しました。

---

## 3. その他のモデルで現在も残るハードコード・制限事項

以下の項目については、下層のアルゴリズム実装の仕様に依存する制限、または露出していないパラメータです。

### ① `Private-GSD`
*   **計測クエリの固定**: 測定するマージナルの次数が `k=2`、ビンの分割数が `bins=[2, 4, 8, 16, 32]` に固定されています。
*   **GA設定**: 遺伝的アルゴリズムの集団サイズ `population_size_muta=50`、`population_size_cross=50`、早期終了フラグ `stop_early=True` が固定されています。
*   **乱数シード**: JAXのPRNGキーが `PRNGKey(0)` にハードコードされており、外部からの乱数シード設定が機能しません。

### ② `RAPpp`
*   **ハイパーパラメータの固定**: `RAPppConfiguration` において、`iterations=[1]`（射影時の繰り返しステップ数）、学習率 `optimizer_learning_rate=[0.003]` などのハイパーパラメータが固定されています。また、マージナルクエリの次数も `K=2` 固定です。

### ③ `PrivSyn`
*   **データセット名**: ラッパー内部で `dataset='adult'` がハードコードされています。
*   **露出不足**: 数値データ前処理手法（`num_preprocess='privtree'`）、稀なカテゴリの閾値（`rare_threshold=0.005`）、一貫性保持ループ回数（`consist_iterations=501`）などがラッパーの引数として露出していません。

### ④ `GEM`
*   **ネットワーク次元**: ジェネレータの隠れ層の次元 `gen_dim=[dim * 2, dim * 2]` が固定されています。
*   **クエリ空間**: WGAN学習時に使用するマージナルの次数が `degree=2`、クエリ数 `workload=100000` に固定されています。


---

## 4. εのみが与えられた場合の動作（δの取り扱い）

ユーザーが $\epsilon$ のみを与えて（すなわち $\delta$ の指定を省略、あるいは純粋な $\epsilon$-DP を期待して $\delta=0$ を明示的に設定して）実行した場合、モデルによって挙動が大きく異なります。

### ① 純粋な $\epsilon$-DP として動作するモデル
*   **`PrivBayes`**
    *   元々 $\epsilon$-DP のアルゴリズムであるため、$\epsilon$ のみで設計通り完全に動作します（$\delta$ は不要です）。

### ② $\delta=0$（純粋な $\epsilon$-DP）の指定でクラッシュするモデル
*   **zCDP 依存モデル (`AIM`, `GEM`, `Private-GSD`, `PrivMRF`, `PrivSyn`, `RAPpp`)**
    *   これらのモデルは、プライバシー予算の変換処理 `cdp_rho` 内で `assert delta > 0` を宣言しています。そのため、$\delta=0$ を明示的に指定すると **`AssertionError` でプログラムが異常終了します**。
    *   $\delta$ を省略した場合（未指定時）は、ラッパーにハードコードされたデフォルト値（例: `delta=1e-5`）が適用され、**$(\epsilon, 10^{-5})$-DP として動作**します。これは純粋な $\epsilon$-DP とは異なる（微小なプライバシー漏洩確率 $\delta$ を許容する）定義です。
*   **`PATE-GAN`**
    *   PATE のノイズスケール計算式 `np.sqrt(2 * np.log(1.25 * 10**self.delta)) / self.epsilon` などの内部処理において $\delta > 0$ であることが前提となっており、$\delta = 0$ では正しく動作しません。省略時はデフォルトの `delta=1e-5` が使用されます。

### ③ εのみを受け取るが、動作は数学的な ε-DP 相当にならないモデル
*   **`DP_MERF`**
    *   ラッパーの引数は `epsilon` のみですが、内部ではその値をそのまま zCDP の $\rho$（`rho = epsilon`）として扱っています。
    *   zCDP で $\rho = \epsilon$ としたモデルを $(\epsilon_{approx}, \delta_{approx})$-DP に再変換すると、通常は指定した $\epsilon$ よりも遥かに大きな値（緩いプライバシー保証）になります。そのため、ユーザーが指定した $\epsilon$ の数値そのものの強度（純粋な $\epsilon$-DP 相当）では動作しません。

