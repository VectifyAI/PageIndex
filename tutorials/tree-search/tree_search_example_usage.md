# `tree_search_example.py` 使い方ガイド

## 1. このスクリプトの概要
`tree_search_example.py` は PageIndex が生成した **文書階層(JSON構造)** を幅優先＋優先度付けで辿りながら、
- LLM を使って「関連ノード」「子を展開すべきノード」を決定するモード
- LLM なしで単純なキーワード一致スコアに基づき探索するオフラインモード

を実行します。最終的に「質問に関連しそうなノード一覧」と「探索トレース」を表示するデモです。

---
## 2. 前提条件
- macOS / zsh（ご利用環境に合わせて調整可）
- Python 3.13 以上（`pyproject.toml` の `requires-python >=3.13`）
- OpenAI APIキー（LLMモード利用時）

Python 3.13 が未インストールなら `pyenv` 等で用意してください（任意）。

---
## 3. リポジトリ取得と環境構築
以下を上から順に zsh へコピペしてください。

```zsh
# (1) リポジトリを取得
git clone https://github.com/sea-turt1e/PageIndex.git
cd PageIndex

# (2) 仮想環境を作成・有効化
python3 -m venv .venv
source .venv/bin/activate  # zsh/macOS

# (3) 依存パッケージをインストール (requirements.txt でも pyproject.toml でもOK)
pip install -r requirements.txt
# もしくは
# pip install .

# (4) OpenAI APIキーを環境変数に設定 (LLM モードを使う場合)
export CHATGPT_API_KEY="あなたのAPIキー"
```

> APIキーが無い場合は後述の `--offline` モードで動作確認できます。

---
## 4. 入力に使う構造 JSON について
デフォルト引数 `--structure results/Attention_is_all_you_need_structure.json` は、`results/` ディレクトリにある PageIndex が抽出した文書階層(JSON)です。形式イメージ:

```json
{
  "structure": [
    {
      "title": "Section 1",
      "summary": "...",
      "start_index": 1,
      "end_index": 3,
      "nodes": [ { "title": "Subsection 1.1", "summary": "...", ... } ]
    },
    ...
  ]
}
```

`tree_search_example.py` 内部ではこれを再帰的に読み込み `TreeNode` オブジェクトへ変換します。

---
## 5. まずは LLM モードで実行
以下コマンドを実行すると、LLM による探索＋トレース出力が得られます。

```zsh
python tutorials/tree-search/tree_search_example.py \
   "What is the main conclusion in this document?" \
  --structure results/Attention_is_all_you_need_structure.json \
  --trace
```

主な出力内容:
- Traversal order (LLM-guided): どのノードをどの順番で候補にしたか
- LLM decision trace: 各ターンの reasoning / relevant_nodes / expand 対象
- Final selected nodes: 最終的に関連ありと判断されたノード一覧

### 追加で調整できる代表的オプション
```text
--model              使用する OpenAI モデル (デフォルト: gpt-4.1-nano-2025-04-14)
--max-depth          子ノードを辿る最大深さ (デフォルト: 3)
--branch-factor      1ノードあたり展開する子の最大数 (デフォルト: 3)
--prompt-node-limit  1ターンでLLMへ渡す候補ノード数 (デフォルト: 5)
--max-turns          LLMターンの上限 (デフォルト: 6)
--max-summary-chars  各ノード summary の最大文字数 (デフォルト: 420)
--verbose            ログを INFO レベルに (思考ログなど)
--no-trace           トレース非表示
```

調整例:
```zsh
# 候補ノードを多めに渡して網羅性を上げたい場合
python tutorials/tree-search/tree_search_example.py "Query" --prompt-node-limit 8 --max-turns 8

# 深い階層も辿りたい場合
python tutorials/tree-search/tree_search_example.py "Query" --max-depth 5

# 要約を短くしてトークン節約
python tutorials/tree-search/tree_search_example.py "Query" --max-summary-chars 200
```

---
## 6. オフライン (キーワードヒューリスティック) モード
APIキー不要で簡易探索を行います。`--offline` を付与するだけです。

```zsh
python tutorials/tree-search/tree_search_example.py \
  "self-attention scaling" \
  --offline \
  --structure results/Attention_is_all_you_need_structure.json \
  --trace
```

仕組み:
- クエリを単語分割し、各ノード (title + summary) に含まれる単語数をスコア化
- スコアが高いノードを表示
- 子ノード展開は、スコア順に `--branch-factor` 件をキューへ追加

出力:
- Keyword heuristic results: スコア付き関連候補
- Traversal order (keyword heuristic)
- Expansion decisions: どの親からどの子を展開したか

---
## 7. 仕組みの簡易内部解説
LLM モード:
1. 幅優先風にキューからノードを取り出し `prompt_node_limit` 件まとめる
2. ノード情報 (ID, depth, pages, title, summary, 先頭子一覧) を整形して LLM にプロンプト送信
3. LLM から JSON (`thinking`, `relevant_nodes`, `expand`) を抽出 (`extract_json` 使用)
4. `expand` 指定ノードの子をキーワード簡易スコアで並び替え、上位 `branch_factor` をキューへ追加
5. `max_turns` もしくはキュー枯渇で終了 → 重複排除して最終ノード一覧表示

オフラインモード:
- キューからノード → キーワードスコア計算 → マッチしたら保存 → 子ノード展開 (スコア順上位 N) の繰り返し。

---
## 8. 典型的なチューニング指針
| 目的 | オプション | 推奨方向 |
|------|------------|----------|
| 網羅性を高めたい | `--max-turns`, `--prompt-node-limit` | 値を増やす |
| 深い章節まで探索 | `--max-depth` | 深さを増やす (ただしコスト増) |
| 子の枝刈り | `--branch-factor` | 小さくして精度重視 / 大きくして広く探索 |
| トークン節約 | `--max-summary-chars` | 小さくする |
| 対話ログ調査 | `--verbose` | 有効化して reasoning をログ確認 |

---
## 9. トラブルシューティング
| 症状 | 原因 | 対処 |
|------|------|------|
| "Set OPENAI_API_KEY..." で終了 | APIキー未設定 | `export OPENAI_API_KEY=...` または `--api-key` 指定 |
| LLM decision trace が空 | モデル応答がパース不能 | `--verbose` で応答確認 / モデル変更 / `--offline` で動作確認 |
| 選択ノードが 0 件 | クエリが抽象的 / ターン不足 | `--max-turns` 増 / クエリを具体化 / `--max-depth` 増 |
| Rate limit エラー | OpenAI 側制限 | 少し待機 / モデルを軽量に変更 |
| 日本語クエリが弱い | summary が英語中心 | クエリを英語併記 (例: "自己注意 (self-attention)") |

---
## 10. 応用: 専門知識やユーザ嗜好の統合
`tutorials/tree-search/README.md` の例のように、プロンプトへ **Expert Preference** を追加するだけでドメイン知識の優先度を反映できます。ベクトル埋め込み再学習は不要で、プロンプト拡張のみで制御可能です。

---
## 11. 独自の構造ファイルを使いたい場合
自分の PDF から PageIndex 構造 JSON を生成した後、`--structure` にそのパスを指定すれば同じ探索ロジックを再利用できます。構造 JSON の各ノードは少なくとも以下キーを持つと扱いやすいです:
- `title`
- `summary` (数百文字程度推奨)
- `start_index` / `end_index` (ページ番号。未知なら -1)
- `nodes`: 子ノード配列

---
## 12. 一連の最小コマンド再掲 (丸ごと実行用)
学習より「まず動かしたい」方向けに最少手順を再掲します。

```zsh
# 1. 取得と移動
git clone https://github.com/sea-turt1e/PageIndex.git
cd PageIndex

# 2. 仮想環境
python3 -m venv .venv
source .venv/bin/activate

# 3. 依存インストール
pip install -r requirements.txt

# 4. (LLM利用時のみ) APIキー
export OPENAI_API_KEY="あなたのAPIキー"

# 5. 実行 (LLM モード)
python tutorials/tree-search/tree_search_example.py "What are the scalability considerations of self-attention?" --trace

# 6. 実行 (オフライン モード)
python tutorials/tree-search/tree_search_example.py "self-attention scaling" --offline --trace
```
