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

## 2. ハードコード・露出不足・バグの調査結果

一部のアルゴリズムでは、下層の実装に可変パラメータやDP機能が存在しているにもかかわらず、ラッパー側で値が固定されている、または設定値が上書きされてしまう箇所が存在します。

### ① `TabDDPM` （DPパラメータの非活性化）
*   **事象**: 下層の `method/TabDDPM/scripts/pretrain_and_finetune.py` 内に Opacus の `PrivacyEngine` を用いた差分プライベート学習が実装されていますが、ラッパー側の [tabddpm.py](../generative_models/tabddpm.py) で以下のように `None` がハードコードされています。
    ```python
    self.diffusion = _finetune(
        ...
        dp_epsilon=None,  # ハードコード
        dp_delta=None,    # ハードコード
        rho_used=None,    # ハードコード
        report_every=False,
    )
    ```
*   **影響**: ラッパー経由では常に非プライベート（DP無効）として動作し、$\epsilon$ の設定を反映できません。

### ② `DP_MERF` （パラメータ強制上書きおよび型誤変換）
*   **強制上書き**: ラッパー側の `dp_merf.py` の初期化引数で `num_features` や `how_many_epochs` を指定して渡しているにもかかわらず、下層の [single_generator_priv_all.py](../method/DP_MERF/single_generator_priv_all.py) の `add_default_params` 関数によって、以下のデフォルト値に強制的に上書きされてしまいます。
    ```python
    def add_default_params(args):
        args.n_features_arg = 2000      # 1000等の指定値が無視され2000に上書き
        args.mini_batch_size_arg = 0.05 
        args.how_many_epochs_arg = 1000 # 100等の指定値が無視され1000に上書き
        return args
    ```
*   **誤変換**: ラッパー [dp_merf.py](../generative_models/dp_merf.py) の `fit()` 内で、本来 zCDP の $\rho$ を渡すべき `merf_main` の引数に、生の $\epsilon$（`self.epsilon`）を直接渡してしまっています（通常は `cdp_rho(epsilon, delta)` 等の変換が必要）。

### ③ `PATE-GAN` （ネットワーク層数・オプティマイザの固定）
*   **ネットワーク構成の固定**: ジェネレータとディスクリミネータの隠れ層次元 `h_dim` が特徴量数と同値（`int(self.nfeatures)`）、ノイズの次元 `z_dim` が特徴量数の1/4（`int(self.nfeatures / 4)`）にハードコードされています。
*   **オプティマイザ設定**: Adamの `beta1=0.5` が固定されています。

### ④ `Private-GSD` （クエリおよびGAパラメータの固定）
*   **計測クエリ**: 測定するマージナルの次数が `k=2`、ビンの分割数が `bins=[2, 4, 8, 16, 32]` に固定されています。
*   **GA設定**: 遺伝的アルゴリズムの集団サイズ `population_size_muta=50`、`population_size_cross=50`、早期終了フラグ `stop_early=True` が固定されています。
*   **乱数シード**: JAXのPRNGキーが `PRNGKey(0)` にハードコードされており、外部からの乱数シード設定が機能しません。

### ⑤ `RAPpp` （最適化設定の固定）
*   **ハイパーパラメータの固定**: `RAPppConfiguration` において、`iterations=[1]`（射影時の繰り返しステップ数）、学習率 `optimizer_learning_rate=[0.003]` などのハイパーパラメータが固定されています。また、マージナルクエリの次数も `K=2` 固定です。

### ⑥ `PrivSyn` （前処理およびチューニングパラメータの隠蔽）
*   **データセット名**: ラッパー内部で `dataset='adult'` がハードコードされています。
*   **露出不足**: 数値データ前処理手法（`num_preprocess='privtree'`）、稀なカテゴリの閾値（`rare_threshold=0.005`）、一貫性保持ループ回数（`consist_iterations=501`）などがラッパーの引数として露出していません。

---

## 3. 修正手段（例：TabDDPMへのDPの適用）

`TabDDPM` で差分プライバシーを制御できるようにするための具体的な修正案です。

1.  **ラッパー `__init__` の拡張**:
    [tabddpm.py](../generative_models/tabddpm.py) で `epsilon` と `delta` 引数を受け取れるようにします。
    ```python
    def __init__(
        self,
        metadata=None,
        steps=50,
        lr=1e-4,
        batch_size=1024,
        num_timesteps=100,
        epsilon=None,
        delta=1e-5,
        device=None,
    ):
        ...
        self.epsilon = epsilon
        self.delta = delta
        
        if self.epsilon is not None:
            self.__name__ = f'TabDDPMEps{self.epsilon}'
        else:
            self.__name__ = 'TabDDPM'
    ```

2.  **`fit()` でのパラメータ伝播とzCDP変換の適用**:
    ```python
    from method.AIM.cdp2adp import cdp_rho

    # fit() 内部
    rho = cdp_rho(self.epsilon, self.delta) if self.epsilon is not None else None
    self.diffusion = _finetune(
        ...
        dp_epsilon=self.epsilon,
        dp_delta=self.delta,
        rho_used=rho,
        report_every=False,
    )
    ```
