# OpenHands Local Agent

OpenAI SDK を Ollama の OpenAI 互換 API に向けて使い、ローカルの `gemma4:e4b` からブラウザ操作とターミナル操作を呼び出せる拡張可能な AI エージェントです。

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
playwright install chromium
Copy-Item .env.example .env
```

Ollama 側でモデルを用意します。モデル名が異なる場合は `.env` の `OLLAMA_MODEL` を変更してください。

```powershell
ollama serve
ollama pull gemma4:e4b
```

## 実行

```powershell
openhands-agent
```

単発指示:

```powershell
openhands-agent "https://example.com を開いてページタイトルを確認して"
```

よく使う直接操作:

```powershell
openhands-agent "open browser"
openhands-agent "open https://example.com"
openhands-agent "OpenAI SDKを検索して"
openhands-agent "terminal: Get-Location"
```

## 軽量化設定

描画が重い場合は `.env` の設定を調整します。既定では表示ブラウザを使いながら、画像・動画・フォントの読み込みを止めて負荷を下げています。

```dotenv
OLLAMA_NUM_CTX=131072
OLLAMA_TEMPERATURE=0.2
BROWSER_LIGHT_MODE=true
BROWSER_BLOCK_RESOURCES=image,media,font
BROWSER_VIEWPORT_WIDTH=1024
BROWSER_VIEWPORT_HEIGHT=720
AGENT_MAX_STEPS=6
AGENT_HISTORY_LIMIT=0
AGENT_TRACE=true
```

ページの見た目確認を優先する場合は `BROWSER_BLOCK_RESOURCES=` を空にしてください。さらに軽くしたい場合は `BROWSER_HEADLESS=true` にするとブラウザ表示なしで操作します。

`gemma4:e4b` のローカルモデル情報では context length が `131072` なので、既定で `OLLAMA_NUM_CTX=131072` を指定しています。`AGENT_HISTORY_LIMIT=0` は会話履歴の自動削除を無効にします。

`AGENT_TRACE=true` にすると、内部推論ではなく、ステップ進行・ツール呼び出し・ツール結果のトレースを表示します。

## 拡張方法

新しい操作を増やすときは `src/openhands_agent/tools/base.py` の `Tool` を継承し、`src/openhands_agent/cli.py` の `build_agent()` で `ToolRegistry` に登録します。

ツールは OpenAI Chat Completions の function tool schema と同じ形で LLM に公開されます。Ollama やモデルの tool calling 対応に差があるため、この実装では function tool call に加えて、JSON 形式の手動ツール指定も受け付けます。
