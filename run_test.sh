#!/bin/bash

# モデル評価スクリプト
# 学習済みモデルのパフォーマンスを評価します

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔹 仮想環境を有効化します..."
source HighwayEnv_Merge/bin/activate

echo "🔹 モデル評価を開始します..."
python src/test_highway.py "$@"

echo "✅ 評価が完了しました"

