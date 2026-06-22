#!/usr/bin/env python3
"""
SpaceMouse Differential IK Node for SO-ARM101

3D マウス (SpaceMouse) の 6 軸入力を手先速度に変換し、
frax ライブラリを用いた差分逆運動学 (Differential IK) により
SO-ARM101 の 5 関節位置指令を計算して ROS2 トピックに配信する。

Subscribes:
    robot_type に応じた関節角トピック (sensor_msgs/JointState)

Publishes:
    robot_type に応じた関節位置指令トピック (sensor_msgs/JointState)

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
    joint_names_mycobot (str)  : myCobot 制御対象の関節名 (カンマ区切り)
                                                             例: "joint1,joint2,joint3,joint4,joint5,joint6"
    robot_type         (str)   : "so101" / "mycobot280"
    joint_state_topic  (str)   : 現在関節角トピック (空文字なら robot_type 既定値)
    joint_command_topic (str)  : 関節指令トピック (空文字なら robot_type 既定値)
    feedback_unit_mode (str)   : フィードバック単位 "range_m100_100" / "rad" / "deg"
    command_unit_mode  (str)   : コマンド単位 "range_m100_100" / "rad" / "deg"
  device_path        (str)   : SpaceMouse デバイスパス (例: /dev/hidraw0). 空の場合は自動検出
  velocity_frame     (str)   : 手先速度指令の基準座標系
                               "world" (デフォルト) : base_link 座標系 (従来動作)
                               "ee"                 : gripper_frame_link 座標系 (直感的操作)
  singularity_avoidance (bool) : 特異点回避を有効にする (デフォルト: True)
                               True  = DLS (Damped Least Squares) 擬似逆行列を使用
                               False = 通常の擬似逆行列 (従来動作)
  variable_damping   (bool)  : 可操作性に基づく可変ダンピングを使用 (singularity_avoidance=True 時)
                               True  = 可操作性が低い時にダンピングを自動増大
                               False = 固定ダンピング damping_lambda を使用
  damping_lambda     (float) : DLS ダンピング係数 (デフォルト: 0.05)
                               大きいほど特異点で安定するが、追従性能が低下する
  manipulability_threshold (float) : 可変ダンピング開始閾値 w0 (デフォルト: 0.04)
                               可操作性 w がこの値を下回ると λ が増大し始める

"""

import threading
import time
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TwistStamped


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _as_bool(value) -> bool:
    """ROS2 パラメータの bool 値を安全に変換する。

    launch ファイルから LaunchConfiguration 経由で渡される値は常に文字列になるため、
    Python の "false" はそのまま bool() にかけると True になる問題を回避する。
    """
    if isinstance(value, bool):
        return value
    return str(value).lower() not in ('false', '0', 'no', 'off')


def _prepare_urdf_for_frax(
    urdf_path: str,
    logger=None,
    keep_joint_names: Optional[List[str]] = None,
) -> str:
    """frax が必要とする前処理 (主鎖抽出 + inertial 補完) を行った URDF パスを返す。"""
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception:
        return urdf_path

    pruned_joint_count = 0
    pruned_link_count = 0

    if keep_joint_names:
        joints = root.findall("joint")
        links = root.findall("link")
        joint_by_name = {j.attrib.get("name", ""): j for j in joints}
        child_to_joint = {}
        for joint in joints:
            child = joint.find("child")
            if child is None:
                continue
            child_name = child.attrib.get("link")
            if child_name:
                child_to_joint[child_name] = joint

        keep_joint_set = set()
        for name in keep_joint_names:
            joint = joint_by_name.get(name)
            if joint is None:
                if logger is not None:
                    logger.warn(f"frax 用URDF抽出: 指定関節 '{name}' が見つかりません。")
                continue
            keep_joint_set.add(joint)

        keep_link_set = set()
        for joint in list(keep_joint_set):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is not None:
                p_link = parent.attrib.get("link")
                if p_link:
                    keep_link_set.add(p_link)
            if child is not None:
                c_link = child.attrib.get("link")
                if c_link:
                    keep_link_set.add(c_link)

        # ルートまでの接続に必要な上流 joint/link を補完する。
        frontier = list(keep_link_set)
        while frontier:
            link_name = frontier.pop()
            parent_joint = child_to_joint.get(link_name)
            if parent_joint is None or parent_joint in keep_joint_set:
                continue
            keep_joint_set.add(parent_joint)
            parent = parent_joint.find("parent")
            child = parent_joint.find("child")
            if parent is not None:
                p_link = parent.attrib.get("link")
                if p_link and p_link not in keep_link_set:
                    keep_link_set.add(p_link)
                    frontier.append(p_link)
            if child is not None:
                c_link = child.attrib.get("link")
                if c_link and c_link not in keep_link_set:
                    keep_link_set.add(c_link)
                    frontier.append(c_link)

        keep_joint_names_set = {
            joint.attrib.get("name", "") for joint in keep_joint_set
        }
        keep_link_names_set = set(keep_link_set)

        for joint in joints:
            if joint.attrib.get("name", "") not in keep_joint_names_set:
                root.remove(joint)
                pruned_joint_count += 1
        for link in links:
            if link.attrib.get("name", "") not in keep_link_names_set:
                root.remove(link)
                pruned_link_count += 1

    added_count = 0
    for link in root.findall("link"):
        if link.find("inertial") is not None:
            continue
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": "0.001"})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": "1e-6",
                "ixy": "0.0",
                "ixz": "0.0",
                "iyy": "1e-6",
                "iyz": "0.0",
                "izz": "1e-6",
            },
        )
        added_count += 1

    if added_count == 0 and pruned_joint_count == 0 and pruned_link_count == 0:
        return urdf_path

    tmp_dir = os.path.join(tempfile.gettempdir(), "three_d_mouse_leader")
    os.makedirs(tmp_dir, exist_ok=True)
    base = os.path.basename(urdf_path)
    if base.endswith(".urdf"):
        out_name = base[:-5] + "_frax.urdf"
    else:
        out_name = base + "_frax.urdf"
    out_path = os.path.join(tmp_dir, out_name)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    if logger is not None:
        details = []
        if pruned_joint_count > 0 or pruned_link_count > 0:
            details.append(
                f"主鎖抽出: {pruned_joint_count} joint, {pruned_link_count} link を除外"
            )
        if added_count > 0:
            details.append(f"inertial 補完: {added_count} link")
        logger.warn(
            "frax 用 URDF を生成: " + ", ".join(details) + f" -> {out_path}"
        )
    return out_path


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
        # 擬似逆行列による最小自乗解 (6 DoF タスク → 5 DoF 関節速度: タスク過剰系)
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
    _DEFAULT_URDF_JOINT_NAMES_MYCOBOT = (
        "joint2_to_joint1",
        "joint3_to_joint2",
        "joint4_to_joint3",
        "joint5_to_joint4",
        "joint6_to_joint5",
        "joint6output_to_joint6",
    )
    _DEFAULT_IO_JOINT_NAMES_MYCOBOT = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )
    # /lekiwi/arm_joint_commands に掲載する名前
    _CMD_JOINT_NAMES_SO101 = (
        "arm_shoulder_pan",
        "arm_shoulder_lift",
        "arm_elbow_flex",
        "arm_wrist_flex",
        "arm_wrist_roll",
        "arm_gripper",
    )
    _FEEDBACK_JOINT_NAMES_SO101 = (
        "arm_shoulder_pan",
        "arm_shoulder_lift",
        "arm_elbow_flex",
        "arm_wrist_flex",
        "arm_wrist_roll",
    )
    _ROBOT_TYPE_SO101 = "so101"
    _ROBOT_TYPE_MYCOBOT280 = "mycobot280"
    _DEFAULT_BASE_FRAME_SO101 = "base_link"
    _DEFAULT_EE_FRAME_SO101 = "gripper_frame_link"
    _DEFAULT_BASE_FRAME_MYCOBOT = "g_base"
    _DEFAULT_EE_FRAME_MYCOBOT = "joint6_flange"
    _VIZ_GRIPPER_JOINT_MYCOBOT = "gripper_controller"
    _DEFAULT_INIT_DEG = (0.0, -45.0, 90.0, -45.0, 0.0)
    _DEFAULT_INIT_DEG_MYCOBOT = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __init__(self):
        super().__init__("spacemouse_ik_node")

        # ── パラメータ宣言 ────────────────────────────────────────────────────
        self.declare_parameter("robot_type", self._ROBOT_TYPE_SO101)
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
        self.declare_parameter(
            "joint_names_mycobot",
            ",".join(self._DEFAULT_URDF_JOINT_NAMES_MYCOBOT),
        )
        self.declare_parameter("joint_state_topic", "")
        self.declare_parameter("joint_command_topic", "")
        self.declare_parameter("feedback_joint_names", "")
        self.declare_parameter("command_joint_names", "")
        self.declare_parameter("feedback_unit_mode", "auto")
        self.declare_parameter("command_unit_mode", "auto")
        self.declare_parameter("base_frame_id", "")
        self.declare_parameter("ee_frame_id", "")
        # SpaceMouse デバイスパス (空 = 自動)
        self.declare_parameter("device_path", "")
        # グリッパー初期角 [degree]
        self.declare_parameter("gripper_init_deg", 0.0)
        # グリッパー開閉角度リミット [degree]
        self.declare_parameter("gripper_min_deg", -10.0)
        self.declare_parameter("gripper_max_deg",  100.0)
        # グリッパー開閉速度 [degree/s] (ボタン押下時)
        self.declare_parameter("gripper_speed_dps", 40.0)
        # EE 座標系選択: True=グリッパ先端, False=手首ロール後の付け根
        self.declare_parameter("use_gripper_tip_ee", True)
        # 手先軌跡表示
        self.declare_parameter("enable_trail", False)
        # 手先球マーカー表示
        self.declare_parameter("enable_ee_sphere", True)
        # 手先姿勢表示 (RGB 軸矢印)
        self.declare_parameter("enable_ee_axes", True)
        # SpaceMouse 入力方向矢印表示
        self.declare_parameter("enable_input_arrows", True)
        # 手先速度指令の基準座標系: "world" = base_link 系, "ee" = 手先座標系
        self.declare_parameter("velocity_frame", "world")
        # タスク空間重み: 位置 (XYZ) と姿勢 (Rx/Ry/Rz) の相対重み
        # 位置を優先したい場合は task_weight_pos を大きく、task_weight_rot を小さくする
        self.declare_parameter("task_weight_pos", 1.0)
        self.declare_parameter("task_weight_rot", 0.3)
        # 特異点回避 (Damped Least Squares)
        self.declare_parameter("singularity_avoidance", True)
        self.declare_parameter("variable_damping", True)
        self.declare_parameter("damping_lambda", 0.05)
        self.declare_parameter("manipulability_threshold", 0.04)
        self.declare_parameter("min_joint_max_velocity", 1.0)
        # 実測角への補正（ドリフト対策のため既定は無効）
        self.declare_parameter("feedback_correction_enabled", False)
        self.declare_parameter("feedback_correction_only_when_commanding", True)
        self.declare_parameter("feedback_correction_alpha", 0.2)
        self.declare_parameter("feedback_correction_deadband_deg", 0.8)
        self.declare_parameter("feedback_correction_max_step_deg", 0.4)
        self.declare_parameter("feedback_correction_command_threshold", 0.01)
        self.declare_parameter("profile_ik_timing", False)
        self.declare_parameter("timing_log_every_n", 120)

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

        raw_robot_type = str(self.get_parameter("robot_type").value).strip().lower()
        if raw_robot_type in ("so101", "so-101"):
            self._robot_type = self._ROBOT_TYPE_SO101
        elif raw_robot_type in ("mycobot", "mycobot280", "mycobot280_m5"):
            self._robot_type = self._ROBOT_TYPE_MYCOBOT280
        else:
            self.get_logger().warn(
                f"robot_type='{raw_robot_type}' は未対応です。'{self._ROBOT_TYPE_SO101}' を使用します。"
            )
            self._robot_type = self._ROBOT_TYPE_SO101

        if self._robot_type == self._ROBOT_TYPE_MYCOBOT280:
            joint_names_str = str(self.get_parameter("joint_names_mycobot").value)
            default_init_deg = self._DEFAULT_INIT_DEG_MYCOBOT
        else:
            joint_names_str = str(self.get_parameter("joint_names_so101").value)
            default_init_deg = self._DEFAULT_INIT_DEG

        self._urdf_joint_names: List[str] = [
            n.strip() for n in joint_names_str.split(",") if n.strip()
        ]
        self._n_joints: int = len(self._urdf_joint_names)
        if self._n_joints == 0:
            raise ValueError("制御対象関節名が空です。joint_names_* を確認してください。")

        init_deg_str = str(self.get_parameter("init_joint_positions").value).strip()
        init_deg_values = [float(v.strip()) for v in init_deg_str.split(",") if v.strip()]
        if len(init_deg_values) != self._n_joints:
            self.get_logger().warn(
                "init_joint_positions の要素数が関節数と一致しないため、ロボット既定値を使用します。"
            )
            init_deg_values = list(default_init_deg)
        self._init_q = np.deg2rad(init_deg_values)

        state_topic_param = str(self.get_parameter("joint_state_topic").value).strip()
        command_topic_param = str(self.get_parameter("joint_command_topic").value).strip()
        if self._robot_type == self._ROBOT_TYPE_MYCOBOT280:
            self._joint_state_topic = state_topic_param or "/mycobot/joint_states"
            self._joint_command_topic = command_topic_param or "/mycobot/joint_commands"
        else:
            self._joint_state_topic = state_topic_param or "/lekiwi/joint_states"
            self._joint_command_topic = command_topic_param or "/lekiwi/arm_joint_commands"

        feedback_names_param = str(self.get_parameter("feedback_joint_names").value).strip()
        if feedback_names_param:
            self._feedback_joint_names = [
                n.strip() for n in feedback_names_param.split(",") if n.strip()
            ]
        elif self._robot_type == self._ROBOT_TYPE_MYCOBOT280:
            self._feedback_joint_names = list(self._DEFAULT_IO_JOINT_NAMES_MYCOBOT)
        else:
            self._feedback_joint_names = list(self._FEEDBACK_JOINT_NAMES_SO101)

        cmd_names_param = str(self.get_parameter("command_joint_names").value).strip()
        if cmd_names_param:
            self._cmd_joint_names = [
                n.strip() for n in cmd_names_param.split(",") if n.strip()
            ]
        elif self._robot_type == self._ROBOT_TYPE_MYCOBOT280:
            self._cmd_joint_names = list(self._DEFAULT_IO_JOINT_NAMES_MYCOBOT)
        else:
            self._cmd_joint_names = list(self._CMD_JOINT_NAMES_SO101)

        feedback_unit_param = str(self.get_parameter("feedback_unit_mode").value).strip().lower()
        command_unit_param = str(self.get_parameter("command_unit_mode").value).strip().lower()
        self._feedback_unit_mode = (
            "rad" if self._robot_type == self._ROBOT_TYPE_MYCOBOT280 else "range_m100_100"
        ) if feedback_unit_param in ("", "auto") else feedback_unit_param
        self._command_unit_mode = (
            "rad" if self._robot_type == self._ROBOT_TYPE_MYCOBOT280 else "range_m100_100"
        ) if command_unit_param in ("", "auto") else command_unit_param

        if self._feedback_unit_mode not in ("range_m100_100", "rad", "deg"):
            raise ValueError(
                f"feedback_unit_mode='{self._feedback_unit_mode}' は未対応です。"
            )
        if self._command_unit_mode not in ("range_m100_100", "rad", "deg"):
            raise ValueError(
                f"command_unit_mode='{self._command_unit_mode}' は未対応です。"
            )
        self._command_has_gripper = (
            self._robot_type == self._ROBOT_TYPE_SO101 and len(self._cmd_joint_names) > self._n_joints
        )

        base_frame_param = str(self.get_parameter("base_frame_id").value).strip()
        ee_frame_param = str(self.get_parameter("ee_frame_id").value).strip()
        if self._robot_type == self._ROBOT_TYPE_MYCOBOT280:
            self._base_frame_id = base_frame_param or self._DEFAULT_BASE_FRAME_MYCOBOT
            self._ee_frame_id = ee_frame_param or self._DEFAULT_EE_FRAME_MYCOBOT
        else:
            self._base_frame_id = base_frame_param or self._DEFAULT_BASE_FRAME_SO101
            self._ee_frame_id = ee_frame_param or self._DEFAULT_EE_FRAME_SO101

        self._gripper_pos: float = math.radians(
            self.get_parameter("gripper_init_deg").value
        )
        self._gripper_min: float = math.radians(self.get_parameter("gripper_min_deg").value)
        self._gripper_max: float = math.radians(self.get_parameter("gripper_max_deg").value)
        self._gripper_speed: float = math.radians(self.get_parameter("gripper_speed_dps").value)
        self._use_gripper_tip_ee: bool = _as_bool(self.get_parameter("use_gripper_tip_ee").value)
        self._enable_trail: bool = _as_bool(self.get_parameter("enable_trail").value)
        self._enable_ee_sphere: bool = _as_bool(self.get_parameter("enable_ee_sphere").value)
        self._enable_ee_axes:   bool = _as_bool(self.get_parameter("enable_ee_axes").value)
        self._enable_input_arrows: bool = _as_bool(self.get_parameter("enable_input_arrows").value)
        self._velocity_frame: str = self.get_parameter("velocity_frame").value.lower()
        self._task_weight_pos: float = self.get_parameter("task_weight_pos").value
        self._task_weight_rot: float = self.get_parameter("task_weight_rot").value
        self._singularity_avoidance: bool = _as_bool(self.get_parameter("singularity_avoidance").value)
        self._variable_damping: bool = _as_bool(self.get_parameter("variable_damping").value)
        self._damping_lambda: float = self.get_parameter("damping_lambda").value
        self._manipulability_threshold: float = self.get_parameter("manipulability_threshold").value
        self._min_joint_max_velocity: float = self.get_parameter("min_joint_max_velocity").value
        self._feedback_correction_enabled: bool = _as_bool(
            self.get_parameter("feedback_correction_enabled").value
        )
        self._feedback_correction_only_when_commanding: bool = _as_bool(
            self.get_parameter("feedback_correction_only_when_commanding").value
        )
        self._feedback_correction_alpha: float = float(
            self.get_parameter("feedback_correction_alpha").value
        )
        self._feedback_correction_deadband_rad: float = math.radians(float(
            self.get_parameter("feedback_correction_deadband_deg").value
        ))
        self._feedback_correction_max_step_rad: float = math.radians(float(
            self.get_parameter("feedback_correction_max_step_deg").value
        ))
        self._feedback_correction_command_threshold: float = float(
            self.get_parameter("feedback_correction_command_threshold").value
        )
        self._profile_ik_timing: bool = _as_bool(
            self.get_parameter("profile_ik_timing").value
        )
        self._timing_log_every_n: int = int(
            self.get_parameter("timing_log_every_n").value
        )
        if self._velocity_frame not in ("world", "ee"):
            self.get_logger().warn(
                f"velocity_frame='{self._velocity_frame}' は無効です。'world' にフォールバックします。"
            )
            self._velocity_frame = "world"
        if not (0.0 <= self._feedback_correction_alpha <= 1.0):
            raise ValueError("feedback_correction_alpha must be in [0.0, 1.0].")
        if self._feedback_correction_deadband_rad < 0.0:
            raise ValueError("feedback_correction_deadband_deg must be >= 0.0.")
        if self._feedback_correction_max_step_rad < 0.0:
            raise ValueError("feedback_correction_max_step_deg must be >= 0.0.")
        if self._feedback_correction_command_threshold < 0.0:
            raise ValueError("feedback_correction_command_threshold must be >= 0.0.")
        if self._timing_log_every_n <= 0:
            raise ValueError("timing_log_every_n must be > 0.")

        # ── 状態変数 ─────────────────────────────────────────────────────────
        self._q: np.ndarray = self._init_q.copy()         # 積分済み関節角 [rad]
        self._q_lock = threading.Lock()
        self._current_q: Optional[np.ndarray] = None      # 最新の実測関節角
        self._sm_state = None                              # SpaceMouse 状態
        self._sm_lock = threading.Lock()
        self._button_states: List[bool] = []               # SpaceMouse ボタン状態
        self._buttons_lock = threading.Lock()
        self._debug_frame_count: int = 0                  # デバッグ出力用カウンタ
        self._manip_debug_count: int = 0                  # 可操作性デバッグ出力用カウンタ
        self._ik_timing_count: int = 0
        self._ik_timing_sum_ms: float = 0.0
        self._ik_timing_min_ms: float = float("inf")
        self._ik_timing_max_ms: float = 0.0

        # ── frax ロボットモデル読み込み ───────────────────────────────────────
        if not self._urdf_path:
            raise ValueError(
                "パラメータ 'urdf_path' が未設定です。"
                "launch ファイルで urdf_path を指定してください。"
            )
        self.get_logger().info(f"URDF を読み込み中: {self._urdf_path}")
        frax_urdf_path = _prepare_urdf_for_frax(
            self._urdf_path,
            self.get_logger(),
            keep_joint_names=(
                list(self._urdf_joint_names)
                if self._robot_type == self._ROBOT_TYPE_MYCOBOT280
                else None
            ),
        )
        try:
            self._robot = _load_robot(frax_urdf_path, self._use_gripper_tip_ee)
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

        # フィードバック/コマンド単位変換用 URDF 関節リミット (ラジアン)
        # lekiwi_teleop_node はデフォルト (use_degrees=False) で RANGE_M100_100 単位を使用する。
        # RANGE_M100_100 の変換式: -100 → lower_limit, +100 → upper_limit
        # q_rad = (q_range + 100) / 200 * (hi - lo) + lo
        self._joint_lo: np.ndarray = np.array(
            [self._robot.joint_lower_limits[i] for i in self._frax_indices]
        )
        self._joint_hi: np.ndarray = np.array(
            [self._robot.joint_upper_limits[i] for i in self._frax_indices]
        )
        raw_joint_max_vel = np.array(
            [self._robot.joint_max_velocities[i] for i in self._frax_indices],
            dtype=np.float64,
        )
        invalid_vel_mask = raw_joint_max_vel <= 1e-6
        if np.any(invalid_vel_mask):
            self.get_logger().warn(
                "URDF の関節速度上限が 0 または未設定のため、"
                f"{np.count_nonzero(invalid_vel_mask)} 関節を "
                f"min_joint_max_velocity={self._min_joint_max_velocity} [rad/s] に置換します。"
            )
        self._joint_max_vel = np.where(
            invalid_vel_mask,
            float(self._min_joint_max_velocity),
            raw_joint_max_vel,
        )

        # JIT コンパイル (初回呼び出しで実施)
        self.get_logger().info("JIT コンパイル中 (初回のみ時間がかかります)...")
        import jax
        import jax.numpy as jnp
        import frax

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_platforms", "cpu")

        # タスク空間重み行列 W = diag(wp, wp, wp, wr, wr, wr)
        # 重み付き擬似逆行列: (W^1/2 J)^+ W^1/2 で位置/姿勢の優先度を調整
        _w_diag = jnp.array([
            self._task_weight_pos, self._task_weight_pos, self._task_weight_pos,
            self._task_weight_rot, self._task_weight_rot, self._task_weight_rot,
        ], dtype=jnp.float64)
        _W_sqrt = jnp.diag(jnp.sqrt(_w_diag))

        @jax.jit
        def _diff_ik_step(q_full, task_vel):
            J, _ = self._robot.velocity_control_matrices(q_full)
            J_ctrl = J[:, jnp.array(self._frax_indices)]
            # 重み付きヤコビアン: W^(1/2) J_ctrl
            J_w = _W_sqrt @ J_ctrl          # (6, n_ctrl)

            if self._singularity_avoidance:
                # SVD ベースの可操作性。
                # rank-deficient なモデルでも det ベースの常時 0 を回避し、
                # 制御可能な特異値成分のみで評価する。
                s = jnp.linalg.svd(J_w, compute_uv=False)
                eps = jnp.asarray(1e-4, dtype=jnp.float64)
                w = jnp.where(
                    jnp.any(s > eps),
                    jnp.prod(jnp.where(s > eps, s, 1.0)),
                    0.0,
                )
                if self._variable_damping:
                    # Nakamura & Hanafusa 型可変ダンピング
                    # w < w0 のとき: λ = λ_max * (1 - w/w0)^2
                    w0 = jnp.asarray(
                        self._manipulability_threshold, dtype=jnp.float64
                    )
                    lam_max = jnp.asarray(
                        self._damping_lambda, dtype=jnp.float64
                    )
                    lam = jnp.where(
                        w < w0,
                        lam_max * (1.0 - w / w0) ** 2,
                        0.0,
                    )
                else:
                    # 固定ダンピング
                    lam = jnp.asarray(self._damping_lambda, dtype=jnp.float64)
                # DLS 擬似逆行列 (right form): J^T (J J^T + λ^2 I)^{-1}
                JJT = J_w @ J_w.T           # (6, 6)
                J_w_pinv = J_w.T @ jnp.linalg.inv(
                    JJT + lam ** 2 * jnp.eye(6, dtype=jnp.float64)
                )
            else:
                # 従来動作: 通常の擬似逆行列
                J_w_pinv = jnp.linalg.pinv(J_w)  # (n_ctrl, 6)

            qd = J_w_pinv @ (_W_sqrt @ task_vel)
            max_vel = jnp.asarray(self._joint_max_vel, dtype=jnp.float64)
            qd = jnp.clip(qd, -max_vel, max_vel)
            return qd

        # EE フレーム制御用: FK で EE 姿勢 (T_ee) を取得する JIT 関数
        @jax.jit
        def _get_ee_transform(q_full):
            """FK: EE のホモジーニアス変換行列 (world frame) を返す。"""
            _, T_ee = self._robot.velocity_control_matrices(q_full)
            return T_ee

        # 可操作性計算用 JIT 関数 (SVD ベース)
        @jax.jit
        def _compute_manipulability(q_full):
            J, _ = self._robot.velocity_control_matrices(q_full)
            J_ctrl = J[:, jnp.array(self._frax_indices)]
            J_w = _W_sqrt @ J_ctrl
            s = jnp.linalg.svd(J_w, compute_uv=False)
            eps = jnp.asarray(1e-4, dtype=jnp.float64)
            return jnp.where(
                jnp.any(s > eps),
                jnp.prod(jnp.where(s > eps, s, 1.0)),
                0.0,
            )

        # ウォームアップ
        _diff_ik_step(
            jnp.array(self._q_full, dtype=jnp.float64),
            jnp.zeros(6, dtype=jnp.float64),
        ).block_until_ready()
        _get_ee_transform(
            jnp.array(self._q_full, dtype=jnp.float64),
        ).block_until_ready()
        _compute_manipulability(
            jnp.array(self._q_full, dtype=jnp.float64),
        ).block_until_ready()
        self._diff_ik_step = _diff_ik_step
        self._get_ee_transform = _get_ee_transform
        self._compute_manipulability = _compute_manipulability
        _sa_mode = (
            f"DLS ({'variable' if self._variable_damping else 'fixed'}, "
            f"λ={self._damping_lambda}, w0={self._manipulability_threshold})"
            if self._singularity_avoidance else "OFF (pinv)"
        )
        self.get_logger().info(
            f"JIT コンパイル完了。robot_type='{self._robot_type}', "
            f"velocity_frame='{self._velocity_frame}', "
            f"task_weight_pos={self._task_weight_pos}, task_weight_rot={self._task_weight_rot}, "
            f"特異点回避={_sa_mode}"
        )

        # ── QoS ───────────────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        self._arm_cmd_pub = self.create_publisher(
            JointState, self._joint_command_topic, qos
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
        # SpaceMouse 生入力 (デバッグ用)
        self._sm_raw_pub = self.create_publisher(
            TwistStamped, "/spacemouse/raw", marker_qos
        )
        # 軌跡バッファ (最大 500 点)
        self._trail_points: list = []
        self._trail_max = 500

        # ── Subscriber ────────────────────────────────────────────────────────
        self._joint_state_sub = self.create_subscription(
            JointState,
            self._joint_state_topic,
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
            f"制御関節: {self._urdf_joint_names}, "
            f"state_topic={self._joint_state_topic}, command_topic={self._joint_command_topic}, "
            f"feedback_unit={self._feedback_unit_mode}, command_unit={self._command_unit_mode}, "
            f"base_frame={self._base_frame_id}, ee_frame={self._ee_frame_id}"
        )

        self.get_logger().info(
            "実測補正: "
            f"enabled={self._feedback_correction_enabled}, "
            f"only_when_commanding={self._feedback_correction_only_when_commanding}, "
            f"alpha={self._feedback_correction_alpha}, "
            f"deadband_deg={math.degrees(self._feedback_correction_deadband_rad):.3f}, "
            f"max_step_deg={math.degrees(self._feedback_correction_max_step_rad):.3f}, "
            f"cmd_threshold={self._feedback_correction_command_threshold}"
        )
        self.get_logger().info(
            f"IK timing profile: enabled={self._profile_ik_timing}, "
            f"log_every_n={self._timing_log_every_n}"
        )

    # ── コールバック ──────────────────────────────────────────────────────────

    def _joint_state_callback(self, msg: JointState):
        """/lekiwi/joint_states から現在関節角を取得する。

        lekiwi_teleop_node はデフォルト (use_degrees=False) で RANGE_M100_100 単位
        (-100 〜 +100) を配信する。frax はラジアンを要求するため、URDF 関節リミットを
        スケールファクタとして変換する。
          q_rad = (q_range + 100) / 200 * (hi - lo) + lo
        """
        if len(msg.position) < self._n_joints:
            return

        if msg.name:
            name_to_pos = dict(zip(msg.name, msg.position))
            q_meas_input = np.array(
                [name_to_pos.get(n, 0.0) for n in self._feedback_joint_names[: self._n_joints]]
            )
        else:
            q_meas_input = np.array(msg.position[: self._n_joints])

        if self._feedback_unit_mode == "range_m100_100":
            q_meas_rad = (
                (q_meas_input + 100.0) / 200.0
                * (self._joint_hi - self._joint_lo)
                + self._joint_lo
            )
        elif self._feedback_unit_mode == "deg":
            q_meas_rad = np.deg2rad(q_meas_input)
        else:
            q_meas_rad = q_meas_input

        with self._q_lock:
            is_first = self._current_q is None
            self._current_q = q_meas_rad
            if is_first:
                # 初回受信時のみ積分値を実測値で初期化
                self._q = q_meas_rad.copy()
                self._q_full[self._frax_indices] = self._q

        if is_first:
            self.get_logger().info(
                f"初回関節角度受信完了。制御準備完了。"
                f"現在角度 [deg]: {np.round(np.rad2deg(q_meas_rad), 2)}"
            )

    def _spacemouse_reader(self):
        """バックグラウンドで SpaceMouse を連続読み取りするスレッド関数。"""
        import pyspacemouse

        def _button_cb(state, buttons):
            """ボタン状態変化コールバック。スレッドセーフに状態を更新する。"""
            with self._buttons_lock:
                self._button_states = [bool(b) for b in buttons]

        device = None
        while rclpy.ok():
            try:
                if self._device_path:
                    device = pyspacemouse.open_by_path(
                        self._device_path, button_callback=_button_cb
                    )
                else:
                    device = pyspacemouse.open(button_callback=_button_cb)
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

    def _apply_feedback_correction(self, task_vel_norm: float) -> None:
        """操作中のみ実測値へ穏やかに寄せて、開ループずれを抑える。"""
        if not self._feedback_correction_enabled:
            return

        if self._feedback_correction_only_when_commanding:
            if task_vel_norm < self._feedback_correction_command_threshold:
                return

        with self._q_lock:
            if self._current_q is None:
                return

            error = self._current_q - self._q
            error = np.where(
                np.abs(error) >= self._feedback_correction_deadband_rad,
                error,
                0.0,
            )
            if not np.any(error):
                return

            correction_step = np.clip(
                error * self._feedback_correction_alpha,
                -self._feedback_correction_max_step_rad,
                self._feedback_correction_max_step_rad,
            )
            self._q = np.clip(self._q + correction_step, self._joint_lo, self._joint_hi)
            self._q_full[self._frax_indices] = self._q

    def _record_ik_timing(self, elapsed_ms: float) -> None:
        self._ik_timing_count += 1
        self._ik_timing_sum_ms += elapsed_ms
        self._ik_timing_min_ms = min(self._ik_timing_min_ms, elapsed_ms)
        self._ik_timing_max_ms = max(self._ik_timing_max_ms, elapsed_ms)

        if self._ik_timing_count % self._timing_log_every_n == 0:
            avg_ms = self._ik_timing_sum_ms / self._ik_timing_count
            self.get_logger().info(
                "[IK timing] "
                f"samples={self._ik_timing_count}, "
                f"avg_ms={avg_ms:.3f}, "
                f"min_ms={self._ik_timing_min_ms:.3f}, "
                f"max_ms={self._ik_timing_max_ms:.3f}"
            )

    def _control_loop(self):
        """メイン制御ループ: SpaceMouse → IK → JointState 配信。"""
        import jax.numpy as jnp

        # 実測関節角を受信していない場合は待機
        if self._current_q is None:
            self.get_logger().warn(
                f"関節角度を受信していません。{self._joint_state_topic} を待機中...",
                throttle_duration_sec=2.0,
            )
            return

        with self._sm_lock:
            sm = self._sm_state

        if sm is None:
            # まだ SpaceMouse が繋がっていない場合は現在位置を保持
            self._publish_commands()
            return

        # ── グリッパー制御 (ボタン押下) ───────────────────────────────────────
        # ボタン 0 + ボタン 1 同時押し → 閉じる (gripper_min_deg でストップ)
        # ボタン 0 のみ              → 開く   (gripper_max_deg でストップ)
        with self._buttons_lock:
            _btns = list(self._button_states)
        _btn0 = _btns[0] if len(_btns) > 0 else False
        _btn1 = _btns[1] if len(_btns) > 1 else False
        _g_step = self._gripper_speed * self._dt
        if _btn0 and _btn1:
            self._gripper_pos = max(self._gripper_min, self._gripper_pos - _g_step)
        elif _btn0:
            self._gripper_pos = min(self._gripper_max, self._gripper_pos + _g_step)

        # ── 手先速度 (task_vel) 計算 ──────────────────────────────────────────
        # SpaceMouse 座標系 → ロボット手先座標系への変換は
        # ゲインパラメータで調整可能。
        # state: x, y, z (並進), roll, pitch, yaw (回転) 各 [-1, 1]
        vx = self._apply_deadzone(sm.x) * self._lin_gains[0]
        vy = self._apply_deadzone(sm.y) * self._lin_gains[1]
        vz = self._apply_deadzone(sm.z) * self._lin_gains[2]
        # pyspacemouse の roll → ロボット Y 軸回転 (wy)
        # pyspacemouse の pitch → ロボット X 軸回転 (wx)
        # pyspacemouse の yaw → ロボット Z 軸回転 (wz, 右ねじ補正で符号反転)
        wx = self._apply_deadzone(sm.pitch) * self._rot_gains[0]
        wy = self._apply_deadzone(sm.roll)  * self._rot_gains[1]
        wz = -self._apply_deadzone(sm.yaw)  * self._rot_gains[2]  # SpaceMouse Yaw 符号反転 (右ねじ正方向に補正)

        task_vel = np.array([vx, vy, vz, wx, wy, wz], dtype=np.float64)

        # ── EE フレーム → ワールドフレーム変換 ───────────────────────────────
        # velocity_frame=="ee" のとき、SpaceMouse 入力を手先座標系として解釈し
        # ワールド座標系の速度指令に変換する (J は world frame で定義されているため)。
        if self._velocity_frame == "ee":
            with self._q_lock:
                q_full_for_fk = self._q_full.copy()
            try:
                T_ee = np.asarray(
                    self._get_ee_transform(
                        jnp.array(q_full_for_fk, dtype=jnp.float64)
                    )
                )
                R_ee = T_ee[:3, :3]  # world ← EE 回転行列
                task_vel_ee = task_vel.copy()           # 変換前 (EE フレーム)
                task_vel = np.concatenate([
                    R_ee @ task_vel[:3],
                    R_ee @ task_vel[3:],
                ])

                # ── デバッグ出力 (30 フレームに 1 回、入力がある場合のみ) ──
                self._debug_frame_count += 1
                if self._debug_frame_count % 30 == 0 and not np.allclose(task_vel_ee[3:], 0.0):
                    np.set_printoptions(precision=3, suppress=True)
                    self.get_logger().info(
                        f"\n[EE mode debug]\n"
                        f"  q_deg       = {np.round(np.rad2deg(q_full_for_fk[self._frax_indices]), 1)}\n"
                        f"  R_ee (world←EE):\n"
                        f"    X_ee in world = {np.round(R_ee[:, 0], 3)}  (EE赤軸)\n"
                        f"    Y_ee in world = {np.round(R_ee[:, 1], 3)}  (EE緑軸)\n"
                        f"    Z_ee in world = {np.round(R_ee[:, 2], 3)}  (EE青軸)\n"
                        f"  omega_EE (指令) = {np.round(task_vel_ee[3:], 4)}\n"
                        f"  omega_world    = {np.round(task_vel[3:], 4)}"
                    )

            except Exception as e:
                self.get_logger().warn(
                    f"EE フレーム変換エラー: {e}", throttle_duration_sec=1.0
                )

        self._apply_feedback_correction(float(np.linalg.norm(task_vel)))

        # SpaceMouse 生入力を配信 (常時)
        raw_msg = TwistStamped()
        raw_msg.header.stamp = self.get_clock().now().to_msg()
        raw_msg.header.frame_id = ""
        raw_msg.twist.linear.x  = float(sm.x)
        raw_msg.twist.linear.y  = float(sm.y)
        raw_msg.twist.linear.z  = float(sm.z)
        raw_msg.twist.angular.x = float(sm.roll)
        raw_msg.twist.angular.y = float(sm.pitch)
        raw_msg.twist.angular.z = float(sm.yaw)
        self._sm_raw_pub.publish(raw_msg)

        # SpaceMouse 入力矢印を配信
        if self._enable_input_arrows:
            # world フレームモード時: base_link 座標系での EE 位置を起点に使用
            arrow_origin: Optional[np.ndarray] = None
            if self._velocity_frame != "ee":
                try:
                    with self._q_lock:
                        _q_for_arrow = self._q_full.copy()
                    T_arrow = np.asarray(
                        self._get_ee_transform(
                            jnp.array(_q_for_arrow, dtype=jnp.float64)
                        )
                    )
                    arrow_origin = T_arrow[:3, 3]
                except Exception:
                    pass
            self._publish_input_arrows(
                self.get_clock().now().to_msg(),
                sm.x, sm.y, sm.z,
                sm.roll, sm.pitch, sm.yaw,
                use_ee_frame=(self._velocity_frame == "ee"),
                ee_origin=arrow_origin,
            )

        # ゼロ速度ならスキップ (位置を保持)
        if np.allclose(task_vel, 0.0):
            self._publish_commands()
            return

        with self._q_lock:
            q_full = self._q_full.copy()

        # ── 差分 IK ───────────────────────────────────────────────────────────
        try:
            ik_t0 = time.perf_counter() if self._profile_ik_timing else None
            qd = np.asarray(
                self._diff_ik_step(
                    jnp.array(q_full, dtype=jnp.float64),
                    jnp.array(task_vel, dtype=jnp.float64),
                )
            )
            if ik_t0 is not None:
                self._record_ik_timing((time.perf_counter() - ik_t0) * 1000.0)
        except Exception as e:
            self.get_logger().warn(
                f"IK 計算エラー: {e}", throttle_duration_sec=1.0
            )
            self._publish_commands()
            return

        # ── 特異点回避デバッグ出力 (60 フレームに 1 回) ──────────────────────
        if self._singularity_avoidance:
            self._manip_debug_count += 1
            if self._manip_debug_count % 60 == 0:
                try:
                    w = float(
                        self._compute_manipulability(
                            jnp.array(q_full, dtype=jnp.float64)
                        )
                    )
                    near = w < self._manipulability_threshold
                    self.get_logger().info(
                        f"[特異点回避] 可操作性 w={w:.4f} "
                        f"(閾値 w0={self._manipulability_threshold:.4f}, "
                        f"{'⚠ 特異点近傍' if near else '正常範囲'})"
                    )
                except Exception:
                    pass

        # ── 積分 (q = q + qd * dt) + 関節リミットクランプ ───────────────────
        with self._q_lock:
            new_q = self._q + qd * self._dt

            # 関節上下限でクランプ (self._joint_lo/hi はラジアン単位でキャッシュ済み)
            new_q = np.clip(new_q, self._joint_lo, self._joint_hi)

            self._q = new_q
            self._q_full[self._frax_indices] = self._q

        self._publish_commands()

    def _publish_commands(self):
        """現在の self._q をロボット設定に応じた関節指令トピックへ配信する。"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._cmd_joint_names)

        with self._q_lock:
            q = self._q.copy()

        if self._command_unit_mode == "range_m100_100":
            q_out = (q - self._joint_lo) / (self._joint_hi - self._joint_lo) * 200.0 - 100.0
        elif self._command_unit_mode == "deg":
            q_out = np.rad2deg(q)
        else:
            q_out = q

        positions = list(q_out)
        if self._command_has_gripper:
            # 既存 so101 系の gripper 指令は degree として扱う
            positions.append(math.degrees(self._gripper_pos))

        while len(positions) < len(self._cmd_joint_names):
            positions.append(0.0)

        msg.position = positions[: len(self._cmd_joint_names)]
        self._arm_cmd_pub.publish(msg)

        # ── 可視化用 /joint_states (URDF 関節名) ──────────────────────────────
        viz_msg = JointState()
        viz_msg.header.stamp = msg.header.stamp
        viz_msg.name = list(self._urdf_joint_names)
        viz_msg.position = list(q)
        if self._robot_type == self._ROBOT_TYPE_MYCOBOT280:
            # adaptive gripper は mimic 連動のため、親関節のみ配信すれば指リンクが追従する
            viz_msg.name.append(self._VIZ_GRIPPER_JOINT_MYCOBOT)
            viz_msg.position.append(self._gripper_pos)
        if self._command_has_gripper:
            viz_msg.name.append("gripper")
            viz_msg.position.append(self._gripper_pos)
        self._viz_joint_state_pub.publish(viz_msg)

        # ── エンドエフェクタ マーカー (FK で位置計算) ─────────────────────────
        self._publish_ee_markers(msg.header.stamp, q)


    def _publish_input_arrows(
        self, stamp, ix: float, iy: float, iz: float,
        iroll: float, ipitch: float, iyaw: float,
        use_ee_frame: bool = False,
        ee_origin: Optional[np.ndarray] = None,
    ):
        """SpaceMouse の入力を矢印として配信する。

        use_ee_frame=True  : gripper_frame_link 座標系の原点に描画 (手先フレームモード)
        use_ee_frame=False : base_link 座標系で描画。ee_origin で起点をグリッパ先端に平行移動
        並進 (XYZ): オレンジの矢印
        回転 (Rx/Ry/Rz): 紫の矢印 (IK 指令と符号一致)
        矢印の長さは入力値に比例 (最大 ARROW_SCALE m)。
        """
        ARROW_SCALE = 0.08   # 入力 1.0 に対応する矢印長 [m]
        SHAFT_D     = 0.004  # 矢印軸径 [m]
        HEAD_D      = 0.010  # 矢印頭径 [m]

        orange = ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.9)
        purple = ColorRGBA(r=0.7, g=0.0,  b=1.0, a=0.9)

        markers = MarkerArray()
        mid = 0

        # world フレームモード時の矢印起点 (base_frame_id 座標系での EE 位置)
        ox = float(ee_origin[0]) if ee_origin is not None else 0.0
        oy = float(ee_origin[1]) if ee_origin is not None else 0.0
        oz = float(ee_origin[2]) if ee_origin is not None else 0.0

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
            tail = Point(x=ox, y=oy, z=oz)
            dx, dy, dz = [c * length for c in direction]
            head = Point(x=ox + dx, y=oy + dy, z=oz + dz)
            m.points = [tail, head]
            m.scale.x = SHAFT_D
            m.scale.y = HEAD_D
            m.scale.z = 0.0
            m.color = color
            return m

        # velocity_frame に合わせて描画フレームを選択
        # ee モード  : ee_frame_id 座標系 (EE 周りの方向)
        # world モード: base_frame_id 座標系 + ee_origin オフセット (位置は EE, 姿勢はワールド固定)
        frame = self._ee_frame_id if use_ee_frame else self._base_frame_id

        # ── 並進矢印 (オレンジ) ───────────────────────────────────────────────
        for val, direction, label in [
            (ix,    (1, 0, 0), "lin_x"),
            (iy,    (0, 1, 0), "lin_y"),
            (iz,    (0, 0, 1), "lin_z"),
        ]:
            markers.markers.append(
                make_arrow(frame, direction, val, orange, label, mid)
            )
            mid += 1

        # ── 回転矢印 (紫, 回転軸方向 = 角速度ベクトル方向) ──────────────────────
        # pyspacemouse roll → ロボット Y 軸回転, pitch → ロボット X 軸回転
        # yaw は IK で符号反転済み (-sm.yaw) のため矢印も反転
        for val, direction, label in [
            (iroll,  (0, 1, 0), "rot_x"),   # roll  → +Y 軸方向
            (ipitch, (1, 0, 0), "rot_y"),   # pitch → +X 軸方向
            (-iyaw,  (0, 0, 1), "rot_z"),   # yaw   → +Z (符号反転で IK と一致)
        ]:
            markers.markers.append(
                make_arrow(frame, direction, val, purple, label, mid)
            )
            mid += 1

        self._input_arrow_pub.publish(markers)

    def _publish_ee_markers(self, stamp, q: np.ndarray):
        """エンドエフェクタマーカーを配信する。
        球は ee_frame_id 座標系の原点に置くことで FK 計算を省略。
        軌跡が必要な場合のみ FK を呼ぶ。
        """
        markers = MarkerArray()

        # ── 球マーカー (ee_frame_id 原点 = EE 位置、FK 計算不要) ───────
        if self._enable_ee_sphere:
            sphere = Marker()
            sphere.header.frame_id = self._ee_frame_id
            sphere.header.stamp = stamp
            sphere.ns = "ee"
            sphere.id = 0
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.02
            sphere.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9)
            markers.markers.append(sphere)
        # ── 軸矢印 (RGB, ee_frame_id 座標系 = EE 姿勢) ────────────
        if self._enable_ee_axes:
            AXIS_LEN  = 0.05   # 軸の長さ [m]
            SHAFT_D   = 0.003  # 軸輸径
            HEAD_D    = 0.008  # 矢印頭径
            axis_defs = [
                (1, (1, 0, 0), ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)),  # X 赤
                (2, (0, 1, 0), ColorRGBA(r=0.1, g=1.0, b=0.1, a=1.0)),  # Y 緑
                (3, (0, 0, 1), ColorRGBA(r=0.1, g=0.1, b=1.0, a=1.0)),  # Z 青
            ]
            for mid, (axis_id, direction, color) in enumerate(axis_defs):
                a = Marker()
                a.header.frame_id = self._ee_frame_id
                a.header.stamp = stamp
                a.ns = "ee_axes"
                a.id = axis_id
                a.type = Marker.ARROW
                a.action = Marker.ADD
                tail = Point(x=0.0, y=0.0, z=0.0)
                head = Point(
                    x=direction[0] * AXIS_LEN,
                    y=direction[1] * AXIS_LEN,
                    z=direction[2] * AXIS_LEN,
                )
                a.points = [tail, head]
                a.scale.x = SHAFT_D
                a.scale.y = HEAD_D
                a.scale.z = 0.0
                a.color = color
                markers.markers.append(a)
        # ── 軌跡 (LINE_STRIP, base_frame_id 座標系で蓄積、FK が必要) ─────────────
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
                trail.header.frame_id = self._base_frame_id
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
