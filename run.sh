# 仮想環境のパス
VENV_PATH="HighwayEnv_Merge/bin/activate"

# 仮想環境が有効か確認（仮想環境が有効なら $VIRTUAL_ENV がセットされる）
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$VENV_PATH" ]; then
        echo "🔹 仮想環境を有効化します..."
        source "$VENV_PATH"
    else
        echo "❌ 仮想環境が見つかりません: $VENV_PATH"
        exit 1
    fi
else
    echo "✅ 仮想環境有効化済み！: $VIRTUAL_ENV"
fi

# Pythonスクリプト実行
python src/train_highway-merge.py
