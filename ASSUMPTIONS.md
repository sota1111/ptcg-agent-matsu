# ASSUMPTIONS — 未確認の仮定リスト（SOT-1670）

最終更新: 2026-07-13。事実確認済みの項目は [`docs/engine-facts.md`](docs/engine-facts.md) にあり、
本書は**一次情報で確認できていない仮定**のみを列挙する。実装がこれらに依存する場合は、
仮定 ID を設計メモ/コードコメントに引用すること。確認でき次第、本書から削除して
engine-facts.md へ昇格する。

| ID | 仮定 | 現時点の根拠（二次情報等） | 確認方法 |
| --- | --- | --- | --- |
| A-1 | 本番対戦の持ち時間は**各プレイヤー合計 約10分**で、使い切るとその時点で敗北 | AICU 紹介記事（2026-07-13 取得, https://note.com/aicu/n/ne9cc5c7b4157）。エンジン側にチェスクロック機構があること自体は事実（Game.h:36,61-62,97-110）だが値と発動はソース外 | Kaggle overview 本文（要ブラウザ/ログイン）または公式 Discussion で確認 |
| A-2 | per-move（1手あたり）の個別時間制限は**ない**（持ち時間の総量制のみ） | エンジンの timer が per-select の減算式チェスクロックであること（Game.h:97-110）からの推定。wmh/ptcg-abc README は「per-move time limit」と表現しており矛盾の可能性あり | 同上 |
| A-3 | 提出環境の CPU/メモリ/プロセス数の制約（探索の並列化可否・世界サンプル数の上限に影響） | 未確認。Kaggle simulation コンペの一般則から単一 CPU・数 GB RAM を仮置き | Kaggle overview / Discussion |
| A-4 | 提出エージェントは**推論時ネットワークアクセス禁止**（外部 API/LLM 呼び出し不可） | 競技ルールの一般則＋過去調査メモ。一次情報未確認 | Kaggle rules ページ本文 |
| A-5 | レーティングは Gaussian 分布ベース（TrueSkill 系）で 24 時間自動対戦、**直近の提出のみ**が最終評価対象（「latest 2 are scored」） | AICU 記事・wmh/ptcg-abc README（いずれも二次情報） | Kaggle overview の Evaluation 節 |
| A-6 | エンジンのカードプール（all_card_data()=1267 種, 2026-07-13 実測）は公式スタンダードレギュレーション（マーク H/I/J, 2026-01-23 以降）に対応する | 公式 regulation ページ（事実）とカード数（事実）は確認済みだが、**対応関係**は未検証 | `data/*_Card_Data.csv` の Expansion/Regulation 列とマーク一覧の突合 |
| A-7 | 時間切れ敗北時の `FinishReason` は既存コードに該当値がなく（State.h:24-31 に Timeout なし）、Kaggle 環境側で `Other=9` または別機構により処理される | ソース内に timerStop 呼び出しが無いという事実からの推定 | 本番提出ログの観察 |
| A-8 | `BattleData.h:58` の `remainingTime[i] = config.timeLimit * 1000`（秒コメントと ×1000 の不整合）は本番評価環境では正しく秒管理される（またはミリ秒管理） | 呼び出し箇所がソース内に無いため挙動未確認 | 本番提出での時間挙動観察 |
| A-9 | 対戦マッチング時の先攻/後攻・デッキ順のシャッフルは毎試合エンジン内 `std::random_device` で決まり、提出側から制御・再現する手段はない | ローカル API には注入手段が無いことを確認済み（Api.h:29-33）。本番環境が別途シード管理している可能性は未確認 | Kaggle Discussion |
| A-10 | SelectContext / LogType / OptionType 等の enum は競技中に**追加**され得るが、既存値の意味は変わらない | `cg/api.py:118,323` の「new elements may be appended」注記（追加があることは事実、非破壊であることは仮定） | 追加時の差分確認 |
