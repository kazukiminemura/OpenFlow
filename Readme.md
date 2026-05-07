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
openhands-agent "terminal: Get-Location"
```

## 拡張方法

新しい操作を増やすときは `src/openhands_agent/tools/base.py` の `Tool` を継承し、`src/openhands_agent/cli.py` の `build_agent()` で `ToolRegistry` に登録します。

ツールは OpenAI Chat Completions の function tool schema と同じ形で LLM に公開されます。Ollama やモデルの tool calling 対応に差があるため、この実装では function tool call に加えて、JSON 形式の手動ツール指定も受け付けます。
