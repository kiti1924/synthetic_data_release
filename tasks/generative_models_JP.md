# 合成データ生成手法ドキュメント＆開発・テストガイド

本ドキュメントは、本リポジトリで利用可能な合成データ生成手法（Generative Models）の概要、ハイパーパラメータ統一の実施内容、および今回作成した小規模な検証用データセットを用いたテスト手順をまとめたものです。

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
*   **内容**: すべてのDP対応モデルラッパーについて、初期化時のデフォルト引数を `epsilon=1.0`, `delta=1e-5` に統一しました。
*   **注記**: Pure $\epsilon$-DP である `PrivBayes` は、インターフェース統一のために `delta` を受け取りますが、内部処理では無視されます。また、`TabDDPM` はデフォルトでDPが有効化されており、非プライベートで動作させたい場合は明示的に `epsilon=None` を指定します。

### ② `TabDDPM` のDP有効化（修正完了）
*   **修正内容**: ラッパー側の `tabddpm.py` において、これまで `None` で固定されていた `dp_epsilon`, `dp_delta`, `rho_used` に対し、指定された `epsilon` と `delta` から `cdp_rho` を用いて計算した値を渡すように修正しました。
*   **効果**: ラッパー経由でも差分プライベートな表形式拡散モデル（Opacusの `PrivacyEngine` を用いた学習）が正常に機能するようになりました。

### ③ `DP_MERF` のパラメータ型誤変換と露出の修正（修正完了）
*   **修正内容**: 
    1. `__init__` に `delta=1e-5` を追加し、他モデルと統一。
    2. `fit()` 内で `cdp_rho(self.epsilon, self.delta)` をインポートして zCDP の $\rho$ を算出し、前処理およびモデル学習の `merf_main` に渡すように変更しました（従来は生の $\epsilon$ を $\rho$ としてそのまま渡していたバグを解消）。
*   *(注: 下層の `single_generator_priv_all.py` 内に、指定された `n_features_arg` や `how_many_epochs_arg` をデフォルト値で強制上書きしてしまう実装が含まれていますが、ラッパーから渡された $\rho$ 値によるDPノイズの加算処理は正しく機能するようになりました。)*

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

ユーザーが $\epsilon$ のみを与えて（すなわち $\delta$ の指定を省略、あるいは純粋な $\epsilon$-DP を期待して $\delta=0$ を明示的に設定して）実行した場合、モデルによって挙動が大きく異なります。

### ① 純粋な $\epsilon$-DP として動作するモデル
*   **`PrivBayes`**
    *   元々 $\epsilon$-DP のアルゴリズムであるため、$\epsilon$ のみで設計通り完全に動作します（$\delta$ は不要です）。

### ② $\delta=0$（純粋な $\epsilon$-DP）の指定でクラッシュするモデル
*   **zCDP 依存モデル (`AIM`, `GEM`, `Private-GSD`, `PrivMRF`, `PrivSyn`, `RAPpp`)**
    *   これらのモデルは、プライバシー予算の変換処理 `cdp_rho` 内で `assert delta > 0` を宣言しています。そのため、$\delta=0$ を明示的に指定すると **`AssertionError` でプログラムが異常終了します**。
    *   $\delta$ を省略した場合（未指定時）は、ラッパーにハードコードされたデフォルト値 `delta=1e-5` が適用され、**$(\epsilon, 10^{-5})$-DP として動作**します。
*   **`PATE-GAN`**
    *   PATE のノイズスケール計算式 `np.sqrt(2 * np.log(1.25 * 10**self.delta)) / self.epsilon` などの内部処理において $\delta > 0$ であることが前提となっており、$\delta = 0$ では正しく動作しません。省略時はデフォルトの `delta=1e-5` が使用されます。

### ③ εのみを受け取るが、動作は数学的な ε-DP 相当にならないモデル
*   **`DP_MERF`**
    *   ラッパーの引数は `epsilon` のみですが、内部ではその値をそのまま zCDP の $\rho$（`rho = epsilon`）として扱っています。指定した $\epsilon$ よりも遥かに大きな値（緩いプライバシー保証）になるため、純粋な $\epsilon$-DP 相当の強度では動作しません。
