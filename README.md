# 3d_mouse_leader
<table>
  <tr>
    <td width="50%">
      <video src="./video/rviz_test.mp4" width="100%" autoplay loop muted playsinline></video>
    </td>
    <td width="50%">
      <video src="./video/control_test.mp4" width="100%" autoplay loop muted playsinline></video>
    </td>
  </tr>
</table>

SpaceMouse (3D マウス) の 6 軸入力を手先速度指令に変換し、
[frax](https://github.com/danielpmorton/frax) ライブラリを用いた**微分逆運動学 (Differential IK)** によって
ロボットの関節位置指令を生成し、ROS2 トピックに配信するパッケージです。

現在は `robot_type` パラメータで次を切り替え可能です。

- `so101` (既定): 既存の `/lekiwi/arm_joint_commands` 連携
- `mycobot280`: `/mycobot/joint_commands` 連携 (`mycobot_jointstate_controller` 向け)

---

## アーキテクチャ

```
SpaceMouse (pyspacemouse)
    ↓ state.x/y/z/roll/pitch/yaw  ([-1, 1])
    × ゲインパラメータ
    ↓ 手先速度指令 task_vel [vx, vy, vz, wx, wy, wz]
frax Manipulator
    - URDF 読み込み (SO-ARM101)
    - EE オフセット: gripper_frame_link (先端) or gripper_link (付け根)
    - Jacobian J (6 × 5)
    - 特異点回避 OFF: qd = pinv(J) @ task_vel
    - 特異点回避 ON:  qd = J^T (J J^T + λ^2 I)^{-1} task_vel  [DLS]
                     └ 可変ダンピング ON 時: λ は可操作度 w に応じて自動増減
    ↓ 関節速度指令 qd
    × dt → 積分 → 関節角 q (+ 関節リミットクランプ)
ROS2 Publisher
    → /lekiwi/arm_joint_commands (sensor_msgs/JointState)
    → /joint_states              (sensor_msgs/JointState, Rviz 可視化用)
    → /ik_debug/ee_markers       (visualization_msgs/MarkerArray, EE球・姿勢軸・軌跡)
    → /ik_debug/input_arrows     (visualization_msgs/MarkerArray, 入力方向矢印)
    → /spacemouse/raw            (geometry_msgs/TwistStamped, SpaceMouse 生入力)
```

---

## 動作確認環境

| 項目 | バージョン / モデル |
|------|-------------------|
| OS | Ubuntu 24.04 |
| ROS2 | Jazzy |
| SpaceMouse | SpaceMouse Compact (P/N: 3DX-600053) |

---

## 依存関係

```bash
# HID ライブラリ (pyspacemouse の前提)
sudo apt-get install libhidapi-dev

pip install frax pyspacemouse
```

> **Note**  
> frax は JAX を使用します。CPU 環境では JAX `0.4.30` + `jax_enable_x64` が推奨です。  
> pyspacemouse 詳細: https://pypi.org/project/pyspacemouse/

### STL メッシュファイル

`meshes/` ディレクトリに SO-ARM100 リポジトリから取得した STL ファイルが含まれています。  
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101/assets)

---

## パッケージのビルド

```bash
cd ~/jazzy_ws
colcon build --packages-select three_d_mouse_leader
source install/setup.bash
```

---

## 使用方法

### robot_type の切り替え

```bash
# 既定 (SO-101)
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py robot_type:=so101

# myCobot 280
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py \
    robot_type:=mycobot280 \
    urdf_path:=$ROS_HOME/src/three_d_mouse_leader/urdf/mycobot_280_m5/mycobot_280_m5.urdf \
    init_joint_positions:="0,0,0,0,0,0"
```

`robot_type:=mycobot280` のときは、既定で以下を使用します。

- feedback topic: `/mycobot/joint_states`
- command topic: `/mycobot/joint_commands`
- feedback unit: `rad`
- command unit: `rad`
- command joint names: `joint1..joint6`
- IK 内部の URDF 関節名は `joint2_to_joint1..joint6output_to_joint6` を使用

### 基本的な起動

```bash
source ~/jazzy_ws/install/setup.bash
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py
```

### 実機なしデバッグ (Rviz + 仮想サーボ)

```bash
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py \
    use_mock_servo:=true \
    init_joint_positions:="0,-45,90,-45,0"
```

仮想サーボノード (`mock_servo_node`) が起動直後に初期関節角を `/lekiwi/joint_states` へ配信し続け、
IK ノードが接続できるまで自動でリトライします。IK ノードからの最初のコマンドを受け取ったら折り返し配信に切り替わります。

### Rviz 可視化オプション付き起動例

```bash
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py \
    use_mock_servo:=true \
    enable_ee_sphere:=true \
    enable_trail:=true \
    enable_input_arrows:=true
```

<!--
### EE 座標系の切り替え

```bash
# グリッパ先端 EE (デフォルト・推奨)
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py use_gripper_tip_ee:=true

# 手首ロール後の付け根 EE
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py use_gripper_tip_ee:=false
```

起動ログに `EE=gripper_frame_link (先端)` または `EE=gripper_link (付け根)` と表示されます。

### ゲインの調整例
-->

```bash
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py \
    lin_gain_x:=0.05 \
    lin_gain_y:=0.05 \
    lin_gain_z:=0.05 \
    rot_gain_pitch:=0.20 \
    deadzone:=0.05
```

### 特異点回避の設定

**デフォルト: 可変ダンピング有効** (`singularity_avoidance:=true`, `variable_damping:=true`)

```bash
# 可変ダンピング ON (デフォルト) — 特異点時のみ自動でダンピングを増大
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py \
    singularity_avoidance:=true \
    variable_damping:=true \
    damping_lambda:=0.05 \
    manipulability_threshold:=0.04
```

```bash
# 固定ダンピング ON — 常に一定のダンピングを加える
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py \
    variable_damping:=false \
    damping_lambda:=0.08
```

> **チューニングの目安:**
> - `damping_lambda` を大きくするほど特異点付近が安定するが、速度の増幅率が低下します
> - `manipulability_threshold` を大きくするほど特異点から遠い姿勢からダンピングが増大始まります
> - 実験中はログ出力される可操作度値を見ながら調整してください

### グリッパーのボタン操作

SpaceMouse の 2 つのボタンでグリッパーを開閉できます。

| ボタン操作 | 動作 |
|-----------|------|
| ボタン **0** のみ押し続ける | グリッパーを**開く** (`gripper_max_deg` に達したら停止) |
| ボタン **0** + ボタン **1** 同時押し続ける | グリッパーを**閉じる** (`gripper_min_deg` に達したら停止) |
| 何も押さない | 現在の開度を保持 |

開閉速度・角度リミットは launch 引数でカスタマイズできます：

```bash
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py \
    gripper_min_deg:=-45.0 \
    gripper_max_deg:=45.0 \
    gripper_speed_dps:=30.0
```

### 実機操作
- [lekiwi用のROS2操作ノード](https://github.com/gomi-kuzu/my_lekiwi_prct)と合わせて使う
```bash
# ターミナル 1: SpaceMouse → IK → トピック配信
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py

# ターミナル 2: ロボット側
ros2 run lekiwi_ros2_teleop lekiwi_teleop_node \
  --ros-args -p robot_port:=/dev/ttyACM0
```

---

## 安全機構

### 起動シーケンスと関節角フィードバック待機

ノードは起動後、以下の順序で安全に動作を開始します：

1. **URDF 読み込み・JIT コンパイル**: frax によるロボットモデル構築
2. **SpaceMouse 接続**: 自動検出または指定パスでデバイスをオープン
3. **関節角フィードバック待機**: `/lekiwi/joint_states` の初回受信を待機
4. **制御開始**: 実測関節角を受信後、IK 計算を開始

**重要**: 実測関節角を受信するまで、SpaceMouse で操作しても指令は配信されません。  
これにより、パラメータの初期角度と実際のロボット姿勢が異なる場合でも、  
誤ったヤコビアン計算による急激な動きを防止します。

### 自由度不足の対処

SO-ARM101 は 関節5軸 （グリッパー除く）ですが、SpaceMouse は 6 DoF 指令を出力します。  
そこで、疑似逆行列により最小二乗解を計算し、実現可能な範囲で手先速度に追従します。

- ロボットの姿勢・特異点により、一部の方向の速度が実現できない場合があります

### 特異点回避 (Damped Least Squares)

通常の疑似逆行列はヤコビアンがランク落ちに近い**特異点**周辺で関節速度が発散し、  
ロボットが急激に動いたり制御が不安定になる場合があります。  
**DLS (Damped Least Squares)** はヤコビアンの逆算にダンピング項を加えることで  
特異点付近でも滑らかな動作を維持します。

#### 仕組み

通常の擬似逆行列と DLS の比較:

| 手法 | 計算式 | 特性 |
|------|--------|------|
| 通常の擬似逆行列 | $\dot{q} = J^+ \dot{p}$ | 特異点付近で関節速度が発散する可能性 |
| DLS (固定ダンピング) | $\dot{q} =  (J^TJ + \lambda^2 I)^{-1} J^T \dot{p}$ | 常に一定のダンピングを加える |
| DLS (可変ダンピング) | $\lambda$ を可操作度 $w$ に応じて自動調整 | 特異点付近のみダンピングを増大 |

#### 可操作度 (Manipulability)

可操作度 $w$ はロボットが特異点にどれだけ近いかを示す指標:

$$w = \sqrt{\det(JJ^T)}$$

- $w$ が大きい → 各方向に動きやすい良い姿勢
- $w$ が小さい → 特異点に近く、一部の方向に動けなくなる
- $w = 0$ → 完全な特異点

#### 可変ダンピング

可操作度 $w$ が閾値 $w_0$ を下回るとダンピング係数 $\lambda$ を自動的に大きくする:

$$\lambda(w) = \begin{cases} \lambda_{\max} \left(1 - \dfrac{w}{w_0}\right)^2 & (w < w_0) \\ 0 & (w \geq w_0) \end{cases}$$

これにより:
- 通常姿勢ではダンピングなし → 速度追従性が高い
- 特異点に近づくほど滑らかにダンピングが増大 → 急激な動作を防止

#### デバッグログ

特異点回避が有効な場合、60 制御サイクルごとに可操作度をログ出力します:

```
[特異点回避] 可操作度 w=0.0823 (閾値 w0=0.0400, 正常範囲)
[特異点回避] 可操作度 w=0.0213 (閾値 w0=0.0400, ⚠ 特異点近傍)
```

---

## パラメータ一覧

### IK ノード (`spacemouse_ik_node`)

| パラメータ              | 型      | デフォルト                          | 説明 |
|------------------------|---------|-------------------------------------|------|
| `urdf_path`            | string  | `<pkg>/urdf/so101_new_calib.urdf`   | SO-ARM101 の URDF ファイルパス |
| `control_frequency`    | float   | `30.0`                              | 制御ループ周波数 [Hz] |
| `lin_gain_x/y/z`       | float   | `0.10`                              | 並進ゲイン [m/s per unit] |
| `rot_gain_roll/pitch/yaw` | float | `0.30`                             | 回転ゲイン [rad/s per unit] |
| `deadzone`             | float   | `0.02`                              | 不感帯 (絶対値がこれ未満なら 0 扱い) |
| `init_joint_positions` | string  | `"0,-45,90,-45,0"`                  | 初期関節角 [degree], カンマ区切り |
| `gripper_init_deg`     | float   | `0.0`                               | グリッパー初期角 [degree] |
| `gripper_min_deg`      | float   | `-10.0`                             | グリッパー最小角度 [degree] (閉方向リミット) |
| `gripper_max_deg`      | float   | `100.0`                             | グリッパー最大角度 [degree] (開方向リミット) |
| `gripper_speed_dps`    | float   | `40.0`                              | ボタン押下時のグリッパー開閉速度 [degree/s] |
| `joint_names_so101`    | string  | `"shoulder_pan,...,wrist_roll"`     | 制御対象 URDF 関節名 (カンマ区切り) |
| `device_path`          | string  | `""`                                | SpaceMouse デバイスパス (空 = 自動検出) |
| `use_gripper_tip_ee`   | bool    | `true`                              | EE をグリッパ先端 (true) / 手首付け根 (false) に設定 |
| `velocity_frame`       | string  | `"world"`                           | 速度指令の基準座標系: `"world"` または `"ee"` (**開発中**) |
| `singularity_avoidance` | bool   | `true`                              | 特異点回避を有効にする (DLS 擬似逆行列を使用) |
| `variable_damping`     | bool    | `true`                              | 可操作度に応じた可変ダンピングを使用 (false = 固定ダンピング) |
| `damping_lambda`       | float   | `0.05`                              | DLS ダンピング係数 λ (大きいほど特異点で安定、追従性低下) |
| `manipulability_threshold` | float | `0.04`                           | 可変ダンピング開始閾値 w₀ (可操作度がこれ以下になると λ が増大) |
| `enable_ee_sphere`     | bool    | `true`                              | Rviz に EE 位置の球マーカー (赤, φ20mm) を表示 |
| `enable_ee_axes`       | bool    | `true`                              | Rviz に EE 姿勢の RGB 軸矢印 (X赤/Y緑/Z青, 50mm) を表示 |
| `enable_trail`         | bool    | `false`                             | Rviz に手先軌跡を表示 |
| `enable_input_arrows`  | bool    | `false`                             | Rviz に SpaceMouse 入力方向矢印を表示 (並進=オレンジ, 回転=紫) |

### 仮想サーボノード (`mock_servo_node`)

| パラメータ              | 型      | デフォルト             | 説明 |
|------------------------|---------|------------------------|------|
| `init_joint_positions` | string  | `"0,-45,90,-45,0,0"`   | 起動直後に配信する初期関節角 [degree] |
| `joint_names_cmd`      | string  | `"arm_shoulder_pan,..."` | 配信する関節名 (カンマ区切り) |

### Launch 引数

上記パラメータに加え、以下の launch 専用引数があります。

| 引数            | デフォルト | 説明 |
|----------------|-----------|------|
| `use_rviz`     | `true`    | Rviz2 を起動するか否か |
| `use_mock_servo` | `false` | 仮想サーボノードを起動するか否か |

---

## トピック

| トピック名                    | 型                                  | 方向 | 説明 |
|------------------------------|-------------------------------------|------|------|
| `/lekiwi/arm_joint_commands` | `sensor_msgs/JointState`            | 配信 | アーム関節位置指令 |
| `/lekiwi/joint_states`       | `sensor_msgs/JointState`            | 購読 | 実測関節角 (フィードバック) |
| `/joint_states`              | `sensor_msgs/JointState`            | 配信 | Rviz 可視化用 (URDF 関節名) |
| `/ik_debug/ee_markers`       | `visualization_msgs/MarkerArray`    | 配信 | EE 位置の球マーカー・姿勢軸・手先軌跡 |
| `/ik_debug/input_arrows`     | `visualization_msgs/MarkerArray`    | 配信 | SpaceMouse 入力方向矢印 |
| `/spacemouse/raw`            | `geometry_msgs/TwistStamped`        | 配信 | SpaceMouse 生入力値 (linear=並進, angular=回転, 各 [-1, 1]) |

---

## Rviz 可視化

`rviz/spacemouse_ik.rviz` に設定済みの Rviz 設定ファイルが同梱されています。

| 表示要素 | 説明 |
|---------|------|
| RobotModel (半透明 α=0.45) | SO-ARM101 の 3D モデル |
| TF | `base_link` / `gripper_link` / `gripper_frame_link` の座標軸 |
| EE 球マーカー (赤, φ20mm) | グリッパ先端の現在位置 (`gripper_frame_link` 原点, `enable_ee_sphere:=true` 時) |
| EE 姿勢軸 (RGB, 50mm) | X=赤/Y=緑/Z=青の軸矢印で手先姿勢を表示 (`enable_ee_axes:=true` 時) |
| 手先軌跡 (黄色線) | 手先の移動履歴 (最大 500 点, `enable_trail:=true` 時) |
| 入力方向矢印 | オレンジ=並進 XYZ (並進速度方向)、紫=回転 Rx/Ry/Rz (角速度ベクトル方向, `enable_input_arrows:=true` 時) |

---

## トラブルシューティング

### SpaceMouse が見つからない

```bash
# 接続デバイス確認
python3 -c "import pyspacemouse; print(pyspacemouse.get_connected_devices())"
# デバイスのパーミッション付与
sudo chmod a+rw /dev/hidraw*
```

### URDF の関節名エラー

```
ValueError: URDF に関節 'xxx' が見つかりません。
```

`joint_names_so101` パラメータを実際の URDF 内の関節名に合わせてください。  
利用可能な関節名はノード起動時のログで確認できます。

