# cabt エンジン仕様 — 一次情報調査結果（SOT-1670）

調査日: 2026-07-13
調査者: solo worker (Claude, SOT-1670)

本書は、エージェント実装の前提となる cabt エンジン仕様（合法手 API・観測の粒度・時間制限・敗北条件）を
**一次情報（エンジン実コード＋公式ページ）から確認した結果**をまとめる。
「確認できた事実」には必ずソース（ファイル名+行 or URL）を付す。確認できなかった項目は仮定として分離し、
一覧は [`ASSUMPTIONS.md`](../ASSUMPTIONS.md) に置く。エンジン挙動と公式ルールの差分は
[`docs/rule-deltas.md`](rule-deltas.md) に記録する（**エンジンが正**）。

## 参照ソースの所在

| ソース | 場所 | 備考 |
| --- | --- | --- |
| Python バインディング | 本リポジトリ `cg/api.py`, `cg/game.py`, `cg/sim.py`, `cg/utils.py` | Kaggle 配布 `sample_submission/cg/` と同一物（`scripts/setup_engine.sh` でコピー）。ライセンス上 gitignore |
| エンジン C++ ソース | `/workspaces/kaggle-ptcg-matsu/data/simulation/extracted/ptcg_engine/ptcgProgram 22/` | Kaggle Simulation データ zip 同梱。以下 `ptcgProgram 22/` と表記。competition-use-only ライセンス・再配布禁止 |
| ローカル実行 | 本リポジトリ `eval/run_match.py` | 2026-07-13 に実測（本書「実測による確認」参照） |

---

## 1. 合法手 API（確認できた事実）

### 1.1 エージェントの契約

- 提出エージェントは `agent(obs_dict: dict) -> list[int]` を実装する。返り値は
  `obs.select.option` への**インデックスのリスト**で、「各要素は `0 <= i < len(obs.select.option)`、
  リスト長は `minCount <= len <= maxCount`（両端含む）、要素の重複禁止」（`main.py:22-30` docstring）。
- **最初の呼び出しのみ** `obs.select == None` で、そのときはデッキ（60 枚のカード ID リスト）を返す
  （`main.py:32-36`）。ローカルの `battle_start(deck0, deck1)` はデッキを直接受け取るため、この
  deck 選択ステップは発生しない（`cg/game.py:19-40`、実測でも初回 select は `IS_FIRST` だった）。
- 選択肢の構造は `Observation.select: SelectData`（`cg/api.py:398-409`）:
  - `type: SelectType`（11 種、`cg/api.py:55-66`）: MAIN / CARD / ATTACHED_CARD /
    CARD_OR_ATTACHED_CARD / ENERGY / SKILL / ATTACK / EVOLVE / COUNT / YES_NO / SPECIAL_CONDITION
  - `context: SelectContext`（49 種、`cg/api.py:68-118`）: MAIN, SETUP_ACTIVE_POKEMON, …, RECOVER_SPECIAL_CONDITION
  - `minCount` / `maxCount`: 選択数の下限/上限。`minCount` は 0 になり得る。`maxCount` は
    `len(option)` を超えない（`cg/api.py:402-403`）
  - `option: list[Option]`: 各選択肢。`Option.type: OptionType`（17 種、`cg/api.py:120-187`）が
    PLAY / ATTACH / EVOLVE / ABILITY / DISCARD / RETREAT / ATTACK / END / CARD / ENERGY … を判別する
  - `remainDamageCounter` / `remainEnergyCost`（`cg/api.py:404-405`）、`deck`（デッキから選ぶときのみ
    非 None、`cg/api.py:407`）、`contextCard` / `effect`（`cg/api.py:408-409`）
- **エンジンの合法手 API が唯一の真実**: `option` に列挙されたものだけが合法手。不正な select は
  ローカル API では例外になる（`cg/game.py:60-65` の `IndexError`、search 系は
  `cg/api.py:609-625` にエラーコード別 `ValueError`: 「minCount <= len(select) <= maxCount」
  「0 <= select elements < len(option)」「Duplicate select elements」）。
- **Enum は競技期間中に追加され得る**: 「Please note that new elements may be appended to the Enum
  during the competition.」（`cg/api.py:118`, `cg/api.py:323`）。属性追加も同様（`cg/api.py:328`）。
  → 未知の enum 値/属性でクラッシュしない実装が必須。

### 1.2 ローカル対戦 API（自前対戦ループ用）

- `battle_start(deck0, deck1) -> (obs_dict, StartData)`（`cg/game.py:19-40`）。60 枚でないと
  `ValueError`（`cg/game.py:31-32`）。デッキ不正時は `battlePtr=None` + `errorPlayer`/`errorType`
  が返る（`ptcgProgram 22/Api.h:12-16, 25-73`）。errorType: 1=不明カードID, 2=同名5枚以上
  （基本エネルギーは除外）, 3=たねポケモン0枚, 4=ACE SPEC 2枚以上（`Api.h:40-73`, 定数は
  `Core.h:11` DECK_SIZE=60, `Core.h:14` PRIZE_SIZE=6, `Core.h:19` DECK_SAME_CARD_MAX=4）。
- `battle_select(list[int]) -> obs_dict`（`cg/game.py:48-66`）、`battle_finish()`（`cg/game.py:43-45`）、
  `visualize_data()`（`cg/game.py:69-75`）。
- 「2エージェント対戦ループ」は提供されない — `current.yourIndex` で手番プレイヤーを判別し
  自前で回す（`eval/run_match.py:27-38` が実装例）。

### 1.3 探索（determinization）API — MCTS 実装で使う

- `search_begin(agent_observation, your_deck, your_prize, opponent_deck, opponent_prize,
  opponent_hand, opponent_active, manual_coin=False) -> SearchState`（`cg/api.py:517-595`）:
  観測から**相手の未知領域（山札・サイド・手札・裏向きバトル場）を予測カード ID で埋めて**
  シミュレーション状態を作る。枚数が実際の `deckCount`/`prize`/`handCount` と一致しないと
  `ValueError`（`cg/api.py:553-570`）。`manual_coin=True` でコインの表裏を選択式にできる
  （`cg/api.py:524, 536`）→ chance node の明示に使える。
- `search_step(search_id, select) -> SearchState`（`cg/api.py:597-627`）で 1 選択進める。
  `search_release(search_id)`（`cg/api.py:633-639`）/ `search_end()`（`cg/api.py:629-631`）でメモリ解放。
- `SearchState.observation.search_begin_input` は None（`cg/api.py:448-450`）。

## 2. 観測の粒度（確認できた事実）

`Observation { select, logs, current }`（`cg/api.py:438-443`）。`current: State`（`cg/api.py:366-379`）。

**見える情報（自分視点）:**

- 自分の手札: `players[yourIndex].hand: list[Card]`（`cg/api.py:359`）
- 両者の公開盤面: active / bench（HP, maxHp, energies, energyCards, tools, preEvolution,
  appearThisTurn — `cg/api.py:338-364`）、discard(トラッシュ)全カード、stadium、特殊状態フラグ
  （poisoned/burned/asleep/paralyzed/confused, `cg/api.py:360-364`）
- 枚数情報: 相手 `handCount`、両者 `deckCount`、prize の枚数（`cg/api.py:355-358`）
- ターン状態: `turn`, `turnActionCount`, `yourIndex`, `firstPlayer`, `supporterPlayed`,
  `stadiumPlayed`, `energyAttached`, `retreated`, `result`（`cg/api.py:368-376`）
- 前回選択以降のイベントログ `logs: list[Log]`（`cg/api.py:441`）。LogType 24 種（`cg/api.py:189-323`）。
  コイン結果（COIN, `cg/api.py:315-317`）や KO・ダメージもここから読む

**見えない情報:**

- 相手の手札の中身: `hand` は「None for the opponent.」（`cg/api.py:359`）。実測でも None を確認
- 山札の中身・順序: `deckCount` のみ（`cg/api.py:355`）。ただしデッキから選ぶ効果の間だけ
  `select.deck` に見える（`cg/api.py:407`）
- サイドの中身: `prize: list[Card | None]` で「None if the card is facedown」（`cg/api.py:357`）
- 相手の裏向きバトルポケモン: `active: list[Pokemon | None]`「None if the card is facedown」（`cg/api.py:352`）
- 相手のドロー/裏向き移動はカード ID なしのログ: DRAW_REVERSE（`cg/api.py:208-209`）、
  MOVE_CARD_REVERSE（`cg/api.py:218-221`）

**カードマスタ:** `all_card_data() -> list[CardData]`（`cg/api.py:495-500`）と
`all_attack() -> list[Attack]`（`cg/api.py:502-507`）で全カード属性（HP, weakness, resistance,
retreatCost, ex/megaEx/tera/aceSpec, evolvesFrom, attacks — `cg/api.py:464-490`）が取れる。
実測: 1267 カード / 1556 ワザ（2026-07-13, 本リポジトリの `cg/` + `data/`）。
→ 評価関数は**カード属性ベース**で作れる（カード ID ハードコード不要）。

## 3. 時間制限（事実と仮定を分離）

**エンジン実コードで確認できた事実:**

- エンジンは**プレイヤーごとの持ち時間（チェスクロック方式）**の仕組みを持つ:
  `GameConfig.timeLimit`「時間制限(秒)、0なら制限無し」（`ptcgProgram 22/Game.h:36`）、
  `std::array<double, 2> remainingTime`（`Game.h:61-62`）、選択ごとに `timerStart()` /
  `timerStop(playerIndex)` で経過時間を差し引き、`remainingTime <= 0` で true を返す（`Game.h:97-110`）。
- **ただし、配布ソース内に `timerStart`/`timerStop` の呼び出し箇所は存在しない**
  （`grep -rn 'timerStart\|timerStop'` の一致は定義 `Game.h:97,103` のみ）。時間切れ処理は
  配布パッケージの外（Kaggle 評価環境側ラッパー）で行われるとみられる。`FinishReason` にも
  Timeout 系の値はない（`ptcgProgram 22/State.h:24-31`）。
- ローカル `BattleStart` は `GameConfig config = {}` で生成し `timeLimit` を設定しない = 0 =
  **ローカル対戦に時間制限はない**（`ptcgProgram 22/Api.h:29-33`）。
- 初期化コードに単位の不整合がある: `remainingTime` は「秒単位」（`Game.h:61`）だが
  `game.remainingTime[i] = config.timeLimit * 1000;`（`ptcgProgram 22/BattleData.h:58`）。
  呼び出し箇所がないため実害は未確認。

**確認できなかった仮定（→ ASSUMPTIONS.md A-1, A-2）:**

- 本番の持ち時間の値。二次情報（AICU の競技紹介記事, https://note.com/aicu/n/ne9cc5c7b4157,
  2026-07-13 取得）は「各プレイヤー最大10分・使い切ると敗北」と報じるが、**Kaggle overview 本文
  （一次情報）では未確認**（後述の通りページが JS レンダリングで本文取得不可）。
- per-move（1手あたり）の追加制限の有無。

**エージェント設計への含意**: 値がいくらであれ持ち時間は**試合全体で共有**の設計なので、
探索は anytime + 残り時間ベースの予算配分にする（親Issue の 80%/20% マージン方針と整合）。

## 4. 勝敗・敗北条件（確認できた事実）

`result` は `State.result`: 「Win player index. -1 if not battle finished.」（`cg/api.py:376`）、
決着時は 0/1=勝者インデックス, 2=引き分け（`cg/api.py:319-321` RESULT ログ, `ptcgProgram 22/State.h:355-366`）。

`FinishReason`（`ptcgProgram 22/State.h:24-31`）: `Prize0=1, Deck0=2, NoActivePokemon=3, Effect=4, Other=9`。

1. **サイド取り切り勝ち**: `prize.empty()` でそのプレイヤーに勝ち点（`State.h:378-383`）
2. **場のポケモン全滅負け**: `active.empty() && bench.empty()` で相手に勝ち点（`State.h:384-387`）
   - 両条件は同一チェック内で採点され、**両者同時成立なら引き分け**（`State.h:388-396`。
     公式ルールのサドンデスとの差分 → `docs/rule-deltas.md` D-1）
3. **山札切れ負け**: 自ターン開始時のドロー前にデッキが空なら負け「デッキ切れ負け」
   （`ptcgProgram 22/GameProc.h:989-993`）
4. **カード効果による勝敗**: `FinishReason::Effect`（`ptcgProgram 22/EffectInstant.h:1683`）
5. **エンジン側の安全弁（公式ルールにない打ち切り → rule-deltas.md D-2）**:
   - 総アクション数 3000 以上で引き分け（`ptcgProgram 22/BattleData.h:66-74`）
   - `turnActionCount >= 10000` で強制ターンエンド、`turn >= 10000` で引き分け
     （`ptcgProgram 22/GameProc.h:757-764`）
6. （時間切れ負けは §3 のとおり配布ソース外。二次情報では「持ち時間を使い切ると敗北」）

RESULT ログの reason コメント（`cg/api.py:319-320`）「1: 0 Prize cards. 2: Start turn with 0 deck
cards. 3: No Pokémon in Active Spot. 4: A card effect.」は上記 `FinishReason` と一致する。

## 5. 乱数と再現性（確認できた事実）

- ローカル `BattleStart` は乱数シードを `std::random_device` から取り、外部から注入する API が
  **ない**（`ptcgProgram 22/Api.h:29-33, 76-78`、`deviceRand = true`）。`AgentStart`（search 系）も
  同様（`Api.h:84-92`）。→ **エンジン内部の乱数（シャッフル・コイン）はローカルでも再現不能**。
  親Issue の「同一シード・同一局面 → 同一着手」の再現性要件は**エージェント側の乱数**に対する
  要件として実装する（エンジン側は固定不可）。探索中のコインは `search_begin(manual_coin=True)`
  で選択式にでき、chance node を決定論化できる（`cg/api.py:524, 536`）。

## 6. 実測による確認（2026-07-13, 本リポジトリで実行）

`venv/bin/python` + `cg/` で 1 試合実行（`eval/run_match.py` 相当のループ）:

- `all_card_data()=1267` / `all_attack()=1556`
- `battle_start` 直後の初回 select は `SelectType.YES_NO(9)` / `SelectContext.IS_FIRST(41)`,
  minCount=maxCount=1, option 2 件（先攻後攻の選択から始まる。deck 選択ステップは無い）
- 相手 `hand` は None、自分 `select.deck` も通常時 None を確認
- ランダム同士で 52 意思決定で決着、`result=1`, RESULT ログ `reason=3`（NoActivePokemon）

## 7. 必読 URL の取得記録（2026-07-13）

| URL | 取得結果 | 要点 |
| --- | --- | --- |
| https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview | **本文取得不可**（JS レンダリングの SPA シェルのみ。meta description「Build an AI Training Agent to play the Pokémon Trading Card Game」） | 本文中の時間制限等は未確認 → ASSUMPTIONS A-1/A-2 |
| 同上（Kaggle 公式 API 経由メタデータ） | 取得成功 | Simulation: 締切 2026-08-16 23:59, merger/new-entrant 締切 2026-08-09, Featured/Knowledge, evaluation_metric=`cabt`, 1日最大5提出, チーム最大5人, 参加チーム4943 |
| https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/overview | **本文取得不可**（同上） | Strategy: 公式 API より 締切 2026-09-13 23:59, merger/new-entrant 締切 2026-09-06, 賞金 $240,000, 1日最大5提出, チーム最大5人, 参加チーム195 |
| https://www.pokemon-card.com/howtoplay/ | 取得成功 | 勝利条件3種（サイド6枚取り切り/相手の場のポケモン全滅/相手がターン開始時にドロー不能）、ターン構造（エネルギー手貼り1回/ワザ1回等）、デッキ60枚・同名4枚まで、弱点2倍・抵抗減算、特殊状態はポケモンチェックで処理 |
| https://www.pokemon-card.com/rules/ | 取得成功 | レギュレーション3形式（スタンダード/エクストラ/殿堂）、フロアルール PDF（通常版 floor-rule_20250919.pdf / 競技版 floor-rule_20260130.pdf）、ペナルティガイドライン、カード別 Q&A データベース |
| https://www.pokemon-card.com/rules/regulation/ | 取得成功 | スタンダードはレギュレーションマーク H/I/J（2026-01-23 以降・2026-01-22 更新）。60枚・同名4枚（基本エネルギー除く）・ACE SPEC 1枚。博士の研究系/ボスの指令系は名前違いでも各1種扱い |

補助的な二次情報（本文取得不可の Kaggle overview の代替として参照。事実認定には使わず仮定の根拠のみ）:

- AICU 競技紹介記事 https://note.com/aicu/n/ne9cc5c7b4157 — 各プレイヤー最大10分・時間切れ敗北、
  Gaussian 分布レーティング・24時間自動対戦、Round1 Strategy 上位8チーム各 $30,000
- wmh/ptcg-abc README https://github.com/wmh/ptcg-abc/blob/main/README.md — 提出物
  `submission.tar.gz`（main.py + deck.csv + cg/）、「5/day; the latest 2 are scored.」

## 8. 確認できなかった仮定

[`ASSUMPTIONS.md`](../ASSUMPTIONS.md) に一覧（時間制限の実値、per-move 制限、提出環境の
CPU/メモリ/ネットワーク制約、レーティング詳細、カードプールとレギュレーションの対応 等）。
