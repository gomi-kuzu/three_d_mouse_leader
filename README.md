# 3d_mouse_leader

SpaceMouse (3D マウス) の 6 軸入力を手先速度指令に変換し、
[frax](https://github.com/danielpmorton/frax) ライブラリを用いた**差分逆運動学 (Differential IK)** によって
SO-ARM101 の 5 軸関節位置指令を生成し、ROS2 トピックに配信するパッケージです。

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
    - qd = pinv(J) @ task_vel
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

### EE 座標系の切り替え

```bash
# グリッパ先端 EE (デフォルト・推奨)
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py use_gripper_tip_ee:=true

# 手首ロール後の付け根 EE
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py use_gripper_tip_ee:=false
```

起動ログに `EE=gripper_frame_link (先端)` または `EE=gripper_link (付け根)` と表示されます。

### ゲインの調整例

```bash
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py \
    lin_gain_x:=0.05 \
    lin_gain_y:=0.05 \
    lin_gain_z:=0.05 \
    rot_gain_pitch:=0.20 \
    deadzone:=0.05
```

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

### データ記録と組み合わせる場合

```bash
# ターミナル 1: SpaceMouse → IK → トピック配信
ros2 launch three_d_mouse_leader spacemouse_ik.launch.py

# ターミナル 2: データレコーダー (lekiwi_ros2_teleop パッケージ)
ros2 launch lekiwi_ros2_teleop lekiwi_record.launch.py \
    launch_teleop:=false \
    dataset_repo_id:=john/lekiwi_pick_place \
    fps:=30 \
    single_task:="Pick and place the bottle cap"
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

SO-ARM101 は 5 DoF（グリッパー除く）ですが、SpaceMouse は 6 DoF 指令を出力します。  
**疑似逆行列（Moore-Penrose pseudoinverse）**により最小ノルム解を計算し、  
実現可能な範囲で手先速度に追従します。

- 手先位置 (3 DoF) は比較的実現されやすい
- 手先姿勢 (3 DoF) の一部は姿勢によって制約される
- ロボットの姿勢・特異点により、一部の方向の速度が実現できない場合があります

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

