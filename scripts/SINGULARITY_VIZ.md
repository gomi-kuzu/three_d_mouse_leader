# 特異点・可操作度 可視化ツール

SO-101 ロボットアームのヤコビアン特異値分解 (SVD) を可視化するユーティリティです。

---

## ファイル

| ファイル | 概要 |
|---|---|
| `visualize_singularity.py` | 静的版。Pan×Lift ヒートマップを事前計算し、スライダーで全関節を操作する。 |
| `visualize_singularity_rt.py` | リアルタイム版。スライダーまたは ROS2 `/joint_states` トピックに追従して 10 Hz で更新する。 |

---

## 前提条件

```bash
# conda 環境に以下が入っていること
conda activate leros_jazzy
# frax, jax, matplotlib, japanize_matplotlib, (rclpy は ROS2 モード時)
```

> **URDF パスの解決順序**
> 1. `AMENT_PREFIX_PATH` が設定されていれば `ament_index_python` でインストール先を参照
> 2. 未設定の場合はスクリプトの親ディレクトリ (`../urdf/`) を自動使用
>
> そのため `source ~/jazzy_ws/install/setup.bash` は **必須ではありません**（ROS2 モード使用時は必要）。

---

## 使い方

### 静的版（スライダー操作）

```bash
source ~/jazzy_ws/install/setup.bash
conda run -n leros_jazzy python3 \
  ~/jazzy_ws/src/three_d_mouse_leader/scripts/visualize_singularity.py
```

### リアルタイム版・スライダーモード

```bash
source ~/jazzy_ws/install/setup.bash
conda run -n leros_jazzy python3 \
  ~/jazzy_ws/src/three_d_mouse_leader/scripts/visualize_singularity_rt.py
```

### リアルタイム版・ROS2 実機リンクモード

ターミナル 1（ロボット or モック起動）:
```bash
source ~/jazzy_ws/install/setup.bash
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py \
  use_mock_servo:=true enable_input_arrows:=true
```

ターミナル 2（可視化）:
```bash
source ~/jazzy_ws/install/setup.bash
conda run -n leros_jazzy python3 \
  ~/jazzy_ws/src/three_d_mouse_leader/scripts/visualize_singularity_rt.py --ros2
```

---

## 表示パネルの読み方

```
┌─────────────────┬─────────────────┬─────────────────┐
│  全特異値 棒グラフ  │  可操作度楕円    │  ヌル空間方向    │
├─────────────────┼─────────────────┼─────────────────┤
│  メトリクス       │  位置部分特異値  │  関節角度        │
└─────────────────┴─────────────────┴─────────────────┘
```

### ① 全特異値 棒グラフ（左上）

ヤコビ行列 `J`（6×5）の特異値 σ₁ ≥ σ₂ ≥ … ≥ σ₅ を棒グラフで表示。

| 色 | 意味 |
|---|---|
| 緑 | 正常（σ ≥ 0.01） |
| 橙 | 注意（0.001 ≤ σ < 0.01） |
| 赤 | 危険・特異点（σ < 0.001） |

赤破線が特異点判定閾値（デフォルト 1×10⁻³）。

---

### ② 可操作度楕円（中央上）

位置部分ヤコビアン `J[:3,:]`（3×5）の特異ベクトルを XZ 平面に投影した楕円。

- **楕円が大きく丸い** → どの方向にも動きやすい（可操作性が高い）
- **楕円が潰れて細い** → 特定方向に動きにくい（特異点に近い）
- 3 本の矢印は楕円の主軸（σ₁/σ₂/σ₃ に対応）

---

### ③ ヌル空間方向（右上）

最小特異値 σ_min に対応する右特異ベクトル（`Vt` の最終行）の各関節成分。

- **棒が大きい関節** = その関節を動かしても EE（エンドエフェクタ）がほとんど動かない方向
- 特異点付近では特定の関節運動が無効化されることを示す

---

### ④ メトリクスパネル（左下）

| 表示 | 意味 |
|---|---|
| `State` | 総合判定（✓正常 / △注意 / ▲危険 / ★特異点!） |
| `w_pos` | 位置可操作度 = `√det(J₃ J₃ᵀ)`。0 に近いほど特異点 |
| `σ_min` | 最小特異値。特異点接近の最も直接的な指標 |
| `κ(J)` | 条件数 = σ₁/σ_min。大きいほど数値的に不安定 |

---

### ⑤ 位置部分特異値（中央下）

`J[:3,:]` のみの特異値。可操作度楕円の 3 主軸の長さに直接対応する。

---

### ⑥ 関節角度パネル（右下）

現在の各関節角度 [deg] とデータソース（スライダー or ROS2）を表示。

---

## 特異点の直感的な理解

```
κ(J) が大きい  →  関節速度の小さな誤差が EE 速度に大きく影響する
σ_min ≈ 0      →  ある関節運動の線形結合でも EE が動かせない姿勢
w_pos ≈ 0      →  体積がゼロの楕円 (=直線か点) に縮退
```

特異点付近で逆運動学を解くと関節速度が発散するため、実運用では `σ_min` や
`w_pos` に下限閾値を設けてダンピング付き擬似逆行列 (DLS) を使用する。
