#!/usr/bin/env python3
"""
SpaceMouse Differential IK Node for SO-ARM101

3D マウス (SpaceMouse) の 6 軸入力を手先速度に変換し、
frax ライブラリを用いた差分逆運動学 (Differential IK) により
SO-ARM101 の 5 関節位置指令を計算して ROS2 トピックに配信する。

Subscribes:
  /lekiwi/joint_states  (sensor_msgs/JointState) : アームの現在関節角度

Publishes:
  /lekiwi/arm_joint_commands  (sensor_msgs/JointState) : アームへの関節位置指令

Parameters:
  urdf_path          (str)   : SO-ARM101 URDF ファイルのパス
  control_frequency  (float) : 制御ループの周波数 [Hz]
  lin_gain_x         (float) : 並進 X 軸ゲイン [m/s per unit]
  lin_gain_y         (float) : 並進 Y 軸ゲイン [m/s per unit]
  lin_gain_z         (float) : 並進 Z 軸ゲイン [m/s per unit]
  rot_gain_roll      (float) : 回転 Roll ゲイン [rad/s per unit]
  rot_gain_pitch     (float) : 回転 Pitch ゲイン [rad/s per unit]
  rot_gain_yaw       (float) : 回転 Yaw ゲイン [rad/s per unit]
  deadzone           (float) : 不感帯 (|value| < deadzone は 0 に丸める)
  use_position_ee    (bool)  : True の場合、手先位置誤差も追従 (False = 速度指令のみ)
  kp_pos             (float) : 位置追従 P ゲイン (use_position_ee=True 時に有効)
  kp_rot             (float) : 姿勢追従 P ゲイン (use_position_ee=True 時に有効)
  init_joint_positions (str) : 初期関節角度 (カンマ区切り, degree)
                               例: "0,90,-90,0,0"
  joint_names_so101  (str)   : 制御対象の関節名 (カンマ区切り)
                               例: "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll"
  device_path        (str)   : SpaceMouse デバイスパス (例: /dev/hidraw0). 空の場合は自動検出

"""

import threading
import time
import math
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


# ─────────────────────────────────────────────────────────────────────────────
# frax 差分 IK ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def _load_robot(urdf_path: str, use_gripper_tip: bool = True):
    """frax Manipulator インスタンスをロードして返す。

    use_gripper_tip=True  : EE = gripper_frame_link (グリッパ先端, 推奨)
    use_gripper_tip=False : EE = gripper_link 原点 (手首ロール後の付け根)
    """
    import jax
    import frax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    ee_offset = None
    if use_gripper_tip:
        # gripper_link → gripper_frame_link の固定変換 (rpy=(0,π,0) → Ry(π))
        ee_offset = np.array([
            [-1.0,  0.0,  0.0, -0.0079      ],
            [ 0.0,  1.0,  0.0, -0.000218121 ],
            [ 0.0,  0.0, -1.0, -0.0981274   ],
            [ 0.0,  0.0,  0.0,  1.0         ],
        ], dtype=np.float64)

    robot = frax.Manipulator(urdf_path, ee_offset=ee_offset)
    return robot


def _build_diff_ik(robot, n_joints: int, kp_pos: float, kp_rot: float):
    """JIT コンパイル済みの差分 IK 関数を返す。

    Args:
        robot       : frax.Manipulator インスタンス
        n_joints    : 制御関節数
        kp_pos      : 位置追従 P ゲイン
        kp_rot      : 姿勢追従 P ゲイン

    Returns:
        diff_ik(q, task_vel) -> joint_vel  (numpy array, shape (n_joints,))
    """
    import jax
    import jax.numpy as jnp

    @jax.jit
    def diff_ik(q, task_vel):
        """
        Args:
            q        : 現在関節角 shape (n_joints,)
            task_vel : 目標手先速度 [vx, vy, vz, wx, wy, wz] shape (6,)
        Returns:
            joint_vel : 関節速度指令 shape (n_joints,)
        """
        J, _ = robot.velocity_control_matrices(q)
        # 擬似逆行列による最小ノルム解 (6 DoF タスク → 5 DoF 関節速度)
        J_pinv = jnp.linalg.pinv(J)
        qd = J_pinv @ task_vel
        # 速度リミットでクランプ
        qd = jnp.clip(
            qd,
            -jnp.asarray(robot.joint_max_velocities),
            jnp.asarray(robot.joint_max_velocities),
        )
        return qd

    return diff_ik


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 ノード
# ─────────────────────────────────────────────────────────────────────────────

class SpaceMouseIKNode(Node):
    """SpaceMouse → 差分 IK → 関節位置指令 ROS2 ノード."""

    # SO-101 の URDF 関節名 (gripper 除く 5 軸)
    _DEFAULT_JOINT_NAMES = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    # /lekiwi/arm_joint_commands に掲載する名前
    _CMD_JOINT_NAMES = (
        "arm_shoulder_pan",
        "arm_shoulder_lift",
        "arm_elbow_flex",
        "arm_wrist_flex",
        "arm_wrist_roll",
        "arm_gripper",
    )
    _DEFAULT_INIT_DEG = (0.0, -45.0, 90.0, -45.0, 0.0)

    def __init__(self):
        super().__init__("spacemouse_ik_node")

        # ── パラメータ宣言 ────────────────────────────────────────────────────
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("control_frequency", 30.0)
        # 並進ゲイン [m/s per unit]
        self.declare_parameter("lin_gain_x", 0.10)
        self.declare_parameter("lin_gain_y", 0.10)
        self.declare_parameter("lin_gain_z", 0.10)
        # 回転ゲイン [rad/s per unit]
        self.declare_parameter("rot_gain_roll",  0.30)
        self.declare_parameter("rot_gain_pitch", 0.30)
        self.declare_parameter("rot_gain_yaw",   0.30)
        # 不感帯
        self.declare_parameter("deadzone", 0.02)
        # 初期関節角度 (degree, カンマ区切り文字列)
        self.declare_parameter(
            "init_joint_positions",
            ",".join(str(d) for d in self._DEFAULT_INIT_DEG),
        )
        # 制御対象 URDF 関節名 (カンマ区切り)
        self.declare_parameter(
            "joint_names_so101",
            ",".join(self._DEFAULT_JOINT_NAMES),
        )
        # SpaceMouse デバイスパス (空 = 自動)
        self.declare_parameter("device_path", "")
        # グリッパー初期角 [degree]
        self.declare_parameter("gripper_init_deg", 0.0)
        # EE 座標系選択: True=グリッパ先端, False=手首ロール後の付け根
        self.declare_parameter("use_gripper_tip_ee", True)
        # 手先軌跡表示
        self.declare_parameter("enable_trail", False)
        # 手先球マーカー表示
        self.declare_parameter("enable_ee_sphere", True)
        # SpaceMouse 入力方向矢印表示
        self.declare_parameter("enable_input_arrows", True)

        # ── パラメータ取得 ────────────────────────────────────────────────────
        self._urdf_path: str = self.get_parameter("urdf_path").value
        self._freq: float = self.get_parameter("control_frequency").value
        self._lin_gains = np.array([
            self.get_parameter("lin_gain_x").value,
            self.get_parameter("lin_gain_y").value,
            self.get_parameter("lin_gain_z").value,
        ])
        self._rot_gains = np.array([
            self.get_parameter("rot_gain_roll").value,
            self.get_parameter("rot_gain_pitch").value,
            self.get_parameter("rot_gain_yaw").value,
        ])
        self._deadzone: float = self.get_parameter("deadzone").value
        self._device_path: str = self.get_parameter("device_path").value

        init_deg_str: str = self.get_parameter("init_joint_positions").value
        self._init_q: np.ndarray = np.deg2rad(
            [float(v.strip()) for v in init_deg_str.split(",")]
        )

        joint_names_str: str = self.get_parameter("joint_names_so101").value
        self._urdf_joint_names: List[str] = [
            n.strip() for n in joint_names_str.split(",")
        ]
        self._n_joints: int = len(self._urdf_joint_names)

        self._gripper_pos: float = math.radians(
            self.get_parameter("gripper_init_deg").value
        )
        self._use_gripper_tip_ee: bool = self.get_parameter("use_gripper_tip_ee").value
        self._enable_trail: bool = self.get_parameter("enable_trail").value
        self._enable_ee_sphere: bool = self.get_parameter("enable_ee_sphere").value
        self._enable_input_arrows: bool = self.get_parameter("enable_input_arrows").value

        # ── 状態変数 ─────────────────────────────────────────────────────────
        self._q: np.ndarray = self._init_q.copy()         # 積分済み関節角 [rad]
        self._q_lock = threading.Lock()
        self._current_q: Optional[np.ndarray] = None      # 最新の実測関節角
        self._sm_state = None                              # SpaceMouse 状態
        self._sm_lock = threading.Lock()

        # ── frax ロボットモデル読み込み ───────────────────────────────────────
        if not self._urdf_path:
            raise ValueError(
                "パラメータ 'urdf_path' が未設定です。"
                "launch ファイルで urdf_path を指定してください。"
            )
        self.get_logger().info(f"URDF を読み込み中: {self._urdf_path}")
        try:
            self._robot = _load_robot(self._urdf_path, self._use_gripper_tip_ee)
        except Exception as e:
            self.get_logger().error(f"frax ロボットモデルの読み込みに失敗: {e}")
            raise

        self.get_logger().info(
            f"ロボット読み込み完了: {self._robot.num_joints} 関節 "
            f"({', '.join(self._robot.joint_names)}) "
            f"EE={'gripper_frame_link (先端)' if self._use_gripper_tip_ee else 'gripper_link (付け根)'}"
        )

        # frax の関節順序マッピング
        self._frax_indices: List[int] = []
        for name in self._urdf_joint_names:
            if name not in self._robot.joint_name_to_index:
                raise ValueError(
                    f"URDF に関節 '{name}' が見つかりません。"
                    f"利用可能な関節名: {self._robot.joint_names}"
                )
            self._frax_indices.append(self._robot.joint_name_to_index[name])

        # frax 全関節角 (制御外は 0)
        self._q_full: np.ndarray = np.zeros(self._robot.num_joints)
        self._q_full[self._frax_indices] = self._q

        # JIT コンパイル (初回呼び出しで実施)
        self.get_logger().info("JIT コンパイル中 (初回のみ時間がかかります)...")
        import jax
        import jax.numpy as jnp
        import frax

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_platforms", "cpu")

        @jax.jit
        def _diff_ik_step(q_full, task_vel):
            J, _ = self._robot.velocity_control_matrices(q_full)
            J_ctrl = J[:, jnp.array(self._frax_indices)]
            J_pinv = jnp.linalg.pinv(J_ctrl)
            qd = J_pinv @ task_vel
            max_vel = jnp.asarray(
                [self._robot.joint_max_velocities[i] for i in self._frax_indices]
            )
            qd = jnp.clip(qd, -max_vel, max_vel)
            return qd

        # ウォームアップ
        _diff_ik_step(
            jnp.array(self._q_full, dtype=jnp.float64),
            jnp.zeros(6, dtype=jnp.float64),
        ).block_until_ready()
        self._diff_ik_step = _diff_ik_step
        self.get_logger().info("JIT コンパイル完了。")

        # ── QoS ───────────────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        self._arm_cmd_pub = self.create_publisher(
            JointState, "/lekiwi/arm_joint_commands", qos
        )
        # Rviz 可視化用: robot_state_publisher に渡す URDF 関節名で配信
        self._viz_joint_state_pub = self.create_publisher(
            JointState, "/joint_states", qos
        )
        # エンドエフェクタ マーカー (球 + 軌跡)
        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._ee_marker_pub = self.create_publisher(
            MarkerArray, "/ik_debug/ee_markers", marker_qos
        )
        self._input_arrow_pub = self.create_publisher(
            MarkerArray, "/ik_debug/input_arrows", marker_qos
        )
        # 軌跡バッファ (最大 500 点)
        self._trail_points: list = []
        self._trail_max = 500

        # ── Subscriber ────────────────────────────────────────────────────────
        self._joint_state_sub = self.create_subscription(
            JointState,
            "/lekiwi/joint_states",
            self._joint_state_callback,
            qos,
        )

        # ── SpaceMouse 読み取りスレッド ────────────────────────────────────────
        self._sm_thread = threading.Thread(
            target=self._spacemouse_reader, daemon=True
        )
        self._sm_thread.start()

        # ── 制御ループタイマー ────────────────────────────────────────────────
        self._dt: float = 1.0 / self._freq
        self._timer = self.create_timer(self._dt, self._control_loop)

        self.get_logger().info(
            f"SpaceMouseIKNode 起動: {self._freq} Hz, "
            f"制御関節: {self._urdf_joint_names}"
        )

    # ── コールバック ──────────────────────────────────────────────────────────

    def _joint_state_callback(self, msg: JointState):
        """/lekiwi/joint_states から現在関節角を取得する。"""
        # 関節名の順序が固定なら直接インデックスで参照
        if len(msg.name) >= self._n_joints:
            target_names = [
                "arm_shoulder_pan",
                "arm_shoulder_lift",
                "arm_elbow_flex",
                "arm_wrist_flex",
                "arm_wrist_roll",
            ]
            name_to_pos = dict(zip(msg.name, msg.position))
            q_meas = np.array(
                [name_to_pos.get(n, 0.0) for n in target_names[: self._n_joints]]
            )
            with self._q_lock:
                is_first = self._current_q is None
                self._current_q = q_meas
                # 積分値を実測値で補正 (ドリフト防止)
                self._q = q_meas.copy()
                self._q_full[self._frax_indices] = self._q
            
            if is_first:
                self.get_logger().info(
                    f"初回関節角度受信完了。制御準備完了。現在角度: {np.rad2deg(q_meas)}"
                )

    def _spacemouse_reader(self):
        """バックグラウンドで SpaceMouse を連続読み取りするスレッド関数。"""
        import pyspacemouse

        device = None
        while rclpy.ok():
            try:
                if self._device_path:
                    device = pyspacemouse.open_by_path(self._device_path)
                else:
                    device = pyspacemouse.open()
                self.get_logger().info(
                    f"SpaceMouse 接続: {device.name}"
                )
                break
            except Exception as e:
                self.get_logger().warn(
                    f"SpaceMouse をオープンできません: {e}. 2 秒後に再試行します。",
                    throttle_duration_sec=10.0,
                )
                time.sleep(2.0)

        if device is None:
            return

        with device:
            while rclpy.ok():
                try:
                    state = device.read()
                    with self._sm_lock:
                        self._sm_state = state
                except Exception as e:
                    self.get_logger().warn(
                        f"SpaceMouse 読み取りエラー: {e}",
                        throttle_duration_sec=5.0,
                    )
                    time.sleep(0.01)

    # ── 制御ループ ────────────────────────────────────────────────────────────

    def _apply_deadzone(self, value: float) -> float:
        return value if abs(value) >= self._deadzone else 0.0

    def _control_loop(self):
        """メイン制御ループ: SpaceMouse → IK → JointState 配信。"""
        import jax.numpy as jnp

        # 実測関節角を受信していない場合は待機
        if self._current_q is None:
            self.get_logger().warn(
                "関節角度を受信していません。/lekiwi/joint_states を待機中...",
                throttle_duration_sec=2.0,
            )
            return

        with self._sm_lock:
            sm = self._sm_state

        if sm is None:
            # まだ SpaceMouse が繋がっていない場合は現在位置を保持
            self._publish_commands()
            return

        # ── 手先速度 (task_vel) 計算 ──────────────────────────────────────────
        # SpaceMouse 座標系 → ロボット手先座標系への変換は
        # ゲインパラメータで調整可能。
        # state: x, y, z (並進), roll, pitch, yaw (回転) 各 [-1, 1]
        vx = self._apply_deadzone(sm.x) * self._lin_gains[0]
        vy = self._apply_deadzone(sm.y) * self._lin_gains[1]
        vz = self._apply_deadzone(sm.z) * self._lin_gains[2]
        wx = self._apply_deadzone(sm.roll)  * self._rot_gains[0]
        wy = self._apply_deadzone(sm.pitch) * self._rot_gains[1]
        wz = self._apply_deadzone(sm.yaw)   * self._rot_gains[2]

        task_vel = np.array([vx, vy, vz, wx, wy, wz], dtype=np.float64)

        # SpaceMouse 入力矢印を配信
        if self._enable_input_arrows:
            self._publish_input_arrows(
                self.get_clock().now().to_msg(),
                sm.x, sm.y, sm.z,
                sm.roll, sm.pitch, sm.yaw,
            )

        # ゼロ速度ならスキップ (位置を保持)
        if np.allclose(task_vel, 0.0):
            self._publish_commands()
            return

        with self._q_lock:
            q_full = self._q_full.copy()

        # ── 差分 IK ───────────────────────────────────────────────────────────
        try:
            qd = np.asarray(
                self._diff_ik_step(
                    jnp.array(q_full, dtype=jnp.float64),
                    jnp.array(task_vel, dtype=jnp.float64),
                )
            )
        except Exception as e:
            self.get_logger().warn(
                f"IK 計算エラー: {e}", throttle_duration_sec=1.0
            )
            self._publish_commands()
            return

        # ── 積分 (q = q + qd * dt) + 関節リミットクランプ ───────────────────
        with self._q_lock:
            new_q = self._q + qd * self._dt

            # 関節上下限でクランプ
            lo = np.array(
                [self._robot.joint_lower_limits[i] for i in self._frax_indices]
            )
            hi = np.array(
                [self._robot.joint_upper_limits[i] for i in self._frax_indices]
            )
            new_q = np.clip(new_q, lo, hi)

            self._q = new_q
            self._q_full[self._frax_indices] = self._q

        self._publish_commands()

    def _publish_commands(self):
        """現在の self._q を /lekiwi/arm_joint_commands に配信する。"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._CMD_JOINT_NAMES)

        with self._q_lock:
            q = self._q.copy()

        # 5 関節 + gripper
        positions = list(q) + [self._gripper_pos]
        # 不足分を 0 で補完 (念のため)
        while len(positions) < len(self._CMD_JOINT_NAMES):
            positions.append(0.0)

        msg.position = positions[: len(self._CMD_JOINT_NAMES)]
        self._arm_cmd_pub.publish(msg)

        # ── 可視化用 /joint_states (URDF 関節名) ──────────────────────────────
        viz_msg = JointState()
        viz_msg.header.stamp = msg.header.stamp
        # 制御5関節 + gripper をまとめて配信 (robot_state_publisher に全関節を渡す)
        viz_msg.name = list(self._urdf_joint_names) + ["gripper"]
        viz_msg.position = list(q) + [self._gripper_pos]
        self._viz_joint_state_pub.publish(viz_msg)

        # ── エンドエフェクタ マーカー (FK で位置計算) ─────────────────────────
        self._publish_ee_markers(msg.header.stamp, q)


    def _publish_input_arrows(
        self, stamp, ix: float, iy: float, iz: float,
        iroll: float, ipitch: float, iyaw: float,
    ):
        """SpaceMouse の入力をグリッパ座標系上の矢印として配信する。

        並進 (XYZ): オレンジの矢印 (gripper_frame_link の各軸方向)
        回転 (Rx/Ry/Rz): 紫の矢印 (各回転軸に直交する接線方向)
            Rx (roll)  → 接線方向 = -Y
            Ry (pitch) → 接線方向 = +X
            Rz (yaw)   → 接線方向 = +Y  ※ただし Z×X = Y
        矢印の長さは入力値に比例 (最大 ARROW_SCALE m)。
        """
        ARROW_SCALE = 0.08   # 入力 1.0 に対応する矢印長 [m]
        SHAFT_D     = 0.004  # 矢印軸径 [m]
        HEAD_D      = 0.010  # 矢印頭径 [m]

        orange = ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.9)
        purple = ColorRGBA(r=0.7, g=0.0,  b=1.0, a=0.9)

        markers = MarkerArray()
        mid = 0

        def make_arrow(frame, direction, value, color, ns, mid):
            """frame 座標系で direction 方向に value スケールの矢印 Marker を返す。"""
            length = float(value) * ARROW_SCALE
            if abs(length) < 1e-4:
                # 非表示 (DELETE)
                m = Marker()
                m.header.frame_id = frame
                m.header.stamp = stamp
                m.ns = ns
                m.id = mid
                m.action = Marker.DELETE
                return m
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = mid
            m.type = Marker.ARROW
            m.action = Marker.ADD
            tail = Point(x=0.0, y=0.0, z=0.0)
            dx, dy, dz = [c * length for c in direction]
            head = Point(x=dx, y=dy, z=dz)
            m.points = [tail, head]
            m.scale.x = SHAFT_D
            m.scale.y = HEAD_D
            m.scale.z = 0.0
            m.color = color
            return m

        frame = "gripper_frame_link"

        # ── 並進矢印 (オレンジ) ───────────────────────────────────────────────
        # gripper_frame_link の X/Y/Z 軸方向にそのまま伸ばす
        for val, direction, label in [
            (ix,    (1, 0, 0), "lin_x"),
            (iy,    (0, 1, 0), "lin_y"),
            (iz,    (0, 0, 1), "lin_z"),
        ]:
            markers.markers.append(
                make_arrow(frame, direction, val, orange, label, mid)
            )
            mid += 1

        # ── 回転矢印 (紫, 接線方向) ───────────────────────────────────────────
        # Rx (roll, X軸回転)  → 接線 = X × Z = -Y
        # Ry (pitch, Y軸回転) → 接線 = Y × Z =  X (右手系: Y×Z = X)
        # Rz (yaw,  Z軸回転)  → 接線 = Z × X =  Y
        for val, direction, label in [
            (iroll,  ( 0, -1, 0), "rot_x"),
            (ipitch, ( 1,  0, 0), "rot_y"),
            (iyaw,   ( 0,  1, 0), "rot_z"),
        ]:
            markers.markers.append(
                make_arrow(frame, direction, val, purple, label, mid)
            )
            mid += 1

        self._input_arrow_pub.publish(markers)

    def _publish_ee_markers(self, stamp, q: np.ndarray):
        """エンドエフェクタマーカーを配信する。
        球は gripper_frame_link 座標系の原点に置くことで FK 計算を省略。
        軌跡が必要な場合のみ FK を呼ぶ。
        """
        markers = MarkerArray()

        # ── 球マーカー (gripper_frame_link 原点 = EE 位置、FK 計算不要) ───────
        if self._enable_ee_sphere:
            sphere = Marker()
            sphere.header.frame_id = "gripper_frame_link"
            sphere.header.stamp = stamp
            sphere.ns = "ee"
            sphere.id = 0
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.02
            sphere.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9)
            markers.markers.append(sphere)

        # ── 軌跡 (LINE_STRIP, base_link 座標系で蓄積、FK が必要) ─────────────
        if self._enable_trail:
            import jax.numpy as jnp
            try:
                with self._q_lock:
                    q_full = self._q_full.copy()
                _, T_ee = self._robot.velocity_control_matrices(
                    jnp.array(q_full, dtype=jnp.float64)
                )
                pos = np.asarray(T_ee)[:3, 3]
            except Exception:
                pos = None

            if pos is not None:
                pt = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
                self._trail_points.append(pt)
                if len(self._trail_points) > self._trail_max:
                    self._trail_points.pop(0)

                trail = Marker()
                trail.header.frame_id = "base_link"
                trail.header.stamp = stamp
                trail.ns = "ee_trail"
                trail.id = 1
                trail.type = Marker.LINE_STRIP
                trail.action = Marker.ADD
                trail.scale.x = 0.003
                trail.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.7)
                trail.points = list(self._trail_points)
                markers.markers.append(trail)

        if markers.markers:
            self._ee_marker_pub.publish(markers)


# ─────────────────────────────────────────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SpaceMouseIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
