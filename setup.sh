#!/bin/bash
# ==============================================================================
# XD2000 (Cray環境) 用 仮想環境セットアップスクリプト
# ※警告: 計算ノードは外部通信不可のため、必ずログインノード(ismxd1/2)で実行すること
# ==============================================================================

echo "=== 1. 既存環境のクリーンアップ ==="
# バージョン不整合（3.12混入など）を防ぐため、古い環境とキャッシュを完全削除
deactivate 2>/dev/null || true
rm -rf .venv
uv cache clean

echo "=== 2. Cray開発環境のロード ==="
# マニュアル推奨: Crayのコンパイラとライブラリ(MPI等)を一括初期化
source $SELECT_PE CRAY
module load cray-python/3.11.5

echo "=== 3. 仮想環境の作成 ==="
# - ロードされたPython3.11.5を明示的に指定
# - --system-site-packagesでCray最適化済みライブラリ(NumPy等)を継承
uv venv --python $(which python3) --system-site-packages .venv
source .venv/bin/activate

echo "=== 4. 通常パッケージのインストール ==="
# scikit-learn, pandasなどをインストール
uv pip install -r requirements.txt

echo "=== 5. mpi4pyの専用ビルド (最重要) ==="
# requirements.txt内でバイナリ(wheel)として入ってしまったmpi4pyを一度削除
uv pip uninstall mpi4py 2>/dev/null || true

# Crayコンパイラ(cc)を明示し、システムの libmpi.so.12 と確実にリンクさせるためソースからビルド
MPICC=cc uv pip install --no-cache-dir --no-binary mpi4py mpi4py

echo "=== 6. インポート・チェック ==="
echo "チェック1: Pythonのパス -> $(which python)"
./.venv/bin/python -c "import sklearn; print('チェック2: sklearn -> OK (', sklearn.__version__, ')')"
./.venv/bin/python -c "import mpi4py; print('チェック3: mpi4py -> OK (', mpi4py.__file__, ')')"

echo "=============================================================================="
echo "✨ セットアップが完了しました！"
echo "引き続き qsub npm_test.sh を実行してジョブを投入してください。"
echo "=============================================================================="