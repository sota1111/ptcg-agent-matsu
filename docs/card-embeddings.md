# カード埋め込みと拡張盤面ベクトルv2（SOT-1676）

Świechowski, Tajmajer, Janusz, *"Improving Hearthstone AI by Combining MCTS and
Supervised Learning Algorithms"* (arXiv:1808.04794) §IV-A/B の cabt 向け再現。

## §IV-A 再現: カード埋め込み

- **方式**: 純 Python word2vec **skip-gram + negative sampling**
  (`train/train_embeddings.py`)。依存追加ゼロ（論文は gensim 等を想定するが、
  本リポジトリは pip-free 方針のため自前実装。縮退ではなく同一アルゴリズム）。
- **コーパス**: エンジンのカードマスタ `cg.api.all_card_data()` / `all_attack()`
  （1267 カード・1556 攻撃）。カード 1 枚 = 1 文書で、内容は
  属性トークン（`type:*`, `energy:*`, `hp:*`(30 刻みバケット), `retreat:*`,
  `is:basic/stage1/stage2/ex/megaex/tera/acespec`, `weak:*`, `resist:*`,
  進化元）＋ 名称トークン ＋ **特性/攻撃の効果テキスト**
  （攻撃はさらに `dmg:*` バケット・`cost:*`・`cost_type:*`）。
- **ハイパーパラメータ**: 次元 **10**・window **10**（いずれも論文 §IV-A の設定）、
  negative 5、epochs 3、lr 0.025 線形減衰、dynamic window（word2vec 慣例）。
- **カードベクトル** = その文書トークンの単語ベクトル平均（論文と同じ集約）。
- **決定性**: 乱数は `agents/rng.py` の Rng（シード注入）のみ。語彙順・初期化・
  window/negative サンプリングすべてシード決定的で、同一シード →
  `train/card_embeddings.json` が **byte-identical** に再現される（検証済み。
  `--seed 99` では別内容になることも確認）。
- **出力** `train/card_embeddings.json`（約 130KB）: `dim` / `cards`（card ID →
  10 次元ベクトル。**カード名は含めない** — マスタのテキストはライセンス制約が
  あるため ID キーのみ）/ `default`（全カード平均。未知カード・裏向きカードの
  フォールバック）/ `meta`（シード等の学習設定）。

再現コマンド:

```bash
venv/bin/python train/train_embeddings.py --out train/card_embeddings.json  # seed 61676
```

### 定性チェック（類似カードが近いこと）

現行デッキのカードについて cosine 最近傍を確認（カード名はライセンス配慮で
ID と属性のみ記載）:

| プローブ | 最近傍3件 | 観察 |
| --- | --- | --- |
| id3（基本エネルギー） | id6 / id5 / id2（すべて基本エネルギー、cos ≥ 0.999） | エネルギー同士が最密クラスタ |
| id721（たね, HP150, 攻撃2） | id514 / id213（たね HP120 攻撃2）ほか | たね・攻撃数・HP帯が揃う |
| id723（1進化 megaEx, HP350） | id84（1進化 ex HP270）/ id868（2進化 megaEx HP350） | 高HP・多取りサイドの進化 ex 同士 |
| id1145（特性持ちトレーナー） | id1188 / id1125 / id1126（特性持ちトレーナー） | ポケモンとトレーナーが分離 |

平均ベクトル表現のため cosine の絶対値は全体に高い（0.99 台）が、相対順序で
カテゴリ（エネルギー / ポケモン段階・HP帯 / トレーナー）が正しく分離している。

## §IV-B 再現: 拡張盤面ベクトル（featurize_v2）

`agents/features.py` の `featurize_v2` / `feature_names_v2`。レイアウト（両
プレイヤー分 + グローバル）:

- 各サイド: v1 の 15 スカラー ＋ **6 スロット**（Active 1 + ベンチ 5 =
  `benchMax`）× (低レベル属性 15 + 埋め込み 10) ＋ 手札（可視フラグ +
  埋め込み平均 10。自分のみ可視、相手はゼロ）＋ トラッシュ（枚数 +
  埋め込み平均 10）
- グローバル: turn / my_turn / スタジアム（有無 + 埋め込み 10）

スロットの低レベル属性: present / hidden(裏向き) / hp / max_hp / damage /
energy / retreat / prize_value / max_attack / basic / stage1 / stage2 / tera /
has_ability / appeared。

**合計 387 次元**（dim=10 時）。論文の 750 次元は Hearthstone の盤面構成
（手札上限 10 枚を個別スロット化等）に由来する。cabt では (a) 盤面枠が
Active 1 + ベンチ 5 で固定、(b) 手札は自分側しか可視でなく枚数上限も大きい
ため個別スロットでなく埋め込み平均で集約、(c) 山札は枚数のみ可視 — という
観測構造に合わせて次元を設計した（固定長・カード単位情報を含む点で論文の
意図を保持）。

- 未知カード ID / 裏向き（None）は CardIndex のニュートラル属性と `default`
  埋め込みへフォールバックし、**クラッシュしない**（テストあり）。
- カード名/ID のハードコードなし（`scripts/lint_hardcoded_cards.py` 0 件。
  `CardEmbeddings` は「学習で得た」ID キーのテーブルであり、手書きの per-card
  重みではない）。

## 既存パイプラインからの選択（v1 と共存）

- value model JSON に `feature_set`（`"v1"` 省略時デフォルト / `"v2"`）を記録。
  `LearnedEvaluator` はそれを読んで抽出器を選ぶ（既存 v1 モデルは無変更で動作）。
- ログ生成: `train/gen_selfplay.py --features v2`
- 学習: `train/train_value.py <logs> --features v2` → モデルに `feature_set: v2`
  が刻まれ、`evaluator="learned"` 指定だけで v2 推論になる。
- 抽出器の解決は `agents/features.py::make_featurizer(feature_set, embeddings)`
  に一元化（埋め込みは既定で `train/card_embeddings.json` を lazy ロード、
  テストでは注入可能）。
