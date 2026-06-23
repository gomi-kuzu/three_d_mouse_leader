#!/usr/bin/env python3
"""
Mock Servo Node (デバッグ用仮想サーボ)

起動時に初期関節角を /lekiwi/joint_states に配信し、
その後は /lekiwi/arm_joint_commands を受け取って指令値を
そのまま現在値として返す。

実機なしで IK ノードと Rviz の動作確認に使用する。

配信単位: RANGE_M100_100 (-100 〜 +100)
  実機の lekiwi_teleop_node (use_degrees=False) と同じ単位で配信する。
  init_joint_positions は [degree] で指定する (IKノードと共通パラメータ)。
  SO-ARM101 の場合 RANGE_M100_100 ≈ degree × 0.95 であるため、
  度値をそのまま RANGE_M100_100 の近似初期値として使用する (~5% 誤差)。

Parameters:
  init_joint_positions (str)  : 初期関節角 [degree], カンマ区切り (IKノードと同じ値を指定)
  joint_names_cmd (str)       : /lekiwi/arm_joint_commands の関節名 (カンマ区切り)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState


class MockServoNode(Node):
    _ROBOT_TYPE_SO101 = "so101"
    _ROBOT_TYPE_MYCOBOT280 = "mycobot280"

    _DEFAULT_CMD_JOINT_NAMES = (
        "arm_shoulder_pan",
        "arm_shoulder_lift",
        "arm_elbow_flex",
        "arm_wrist_flex",
        "arm_wrist_roll",
        "arm_gripper",
    )
    _DEFAULT_CMD_JOINT_NAMES_MYCOBOT = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )
    # 初期値は degree 単位で指定 (RANGE_M100_100 の近似値として使用)
    _DEFAULT_INIT_RANGE = (0.0, -45.0, 90.0, -45.0, 0.0, 0.0)

    def __init__(self):
        super().__init__("mock_servo_node")

        self.declare_parameter("robot_type", self._ROBOT_TYPE_SO101)
        self.declare_parameter("joint_state_topic", "")
        self.declare_parameter("joint_command_topic", "")
        self.declare_parameter("feedback_unit_mode", "auto")

        self.declare_parameter(
            "init_joint_positions",
            ",".join(str(d) for d in self._DEFAULT_INIT_RANGE),
        )
        self.declare_parameter(
            "joint_names_cmd",
            "",
        )

        raw_robot_type = str(self.get_parameter("robot_type").value).strip().lower()
        if raw_robot_type in ("mycobot", "mycobot280", "mycobot280_m5"):
            self._robot_type = self._ROBOT_TYPE_MYCOBOT280
        else:
            self._robot_type = self._ROBOT_TYPE_SO101

        if self._robot_type == self._ROBOT_TYPE_MYCOBOT280:
            default_state_topic = "/mycobot/joint_states"
            default_cmd_topic = "/mycobot/joint_commands"
            default_unit_mode = "rad"
            default_joint_names = self._DEFAULT_CMD_JOINT_NAMES_MYCOBOT
        else:
            default_state_topic = "/lekiwi/joint_states"
            default_cmd_topic = "/lekiwi/arm_joint_commands"
            default_unit_mode = "range_m100_100"
            default_joint_names = self._DEFAULT_CMD_JOINT_NAMES

        state_topic_param = str(self.get_parameter("joint_state_topic").value).strip()
        cmd_topic_param = str(self.get_parameter("joint_command_topic").value).strip()
        self._state_topic = state_topic_param or default_state_topic
        self._cmd_topic = cmd_topic_param or default_cmd_topic

        unit_mode_param = str(self.get_parameter("feedback_unit_mode").value).strip().lower()
        self._feedback_unit_mode = (
            default_unit_mode if unit_mode_param in ("", "auto") else unit_mode_param
        )
        if self._feedback_unit_mode not in ("range_m100_100", "rad", "deg"):
            self.get_logger().warn(
                f"feedback_unit_mode='{self._feedback_unit_mode}' は未対応です。'{default_unit_mode}' を使用します。"
            )
            self._feedback_unit_mode = default_unit_mode

        init_str: str = self.get_parameter("init_joint_positions").value
        init_deg_positions = [float(v.strip()) for v in init_str.split(",") if v.strip()]

        names_str: str = self.get_parameter("joint_names_cmd").value
        parsed_names = [n.strip() for n in names_str.split(",") if n.strip()]
        self._joint_names = parsed_names if parsed_names else list(default_joint_names)

        if self._feedback_unit_mode == "rad":
            self._init_positions = [math.radians(v) for v in init_deg_positions]
        elif self._feedback_unit_mode == "deg":
            self._init_positions = list(init_deg_positions)
        else:
            # SO-ARM101 の既存挙動: degree を RANGE_M100_100 の近似値として扱う
            self._init_positions = list(init_deg_positions)

        # 不足分を 0 で補完
        while len(self._init_positions) < len(self._joint_names):
            self._init_positions.append(0.0)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub = self.create_publisher(JointState, self._state_topic, qos)
        self._sub = self.create_subscription(
            JointState,
            self._cmd_topic,
            self._cmd_callback,
            qos,
        )

        # コマンド受信前は初期値を定期配信し続ける (IKノードの接続待ちに対応)
        self._received_cmd = False
        self._init_timer = self.create_timer(0.1, self._publish_initial)

        self.get_logger().info(
            f"MockServoNode 起動: robot_type={self._robot_type}, "
            f"state_topic={self._state_topic}, cmd_topic={self._cmd_topic}, "
            f"unit={self._feedback_unit_mode}, 初期値={[round(p, 3) for p in self._init_positions]}"
        )

    def _publish_initial(self):
        """コマンド未受信の間は初期値を配信し続ける。"""
        if not self._received_cmd:
            self._publish(self._joint_names, self._init_positions)

    def _cmd_callback(self, msg: JointState):
        if not self._received_cmd:
            self._received_cmd = True
            self._init_timer.cancel()
            self.get_logger().info("IKノードからコマンド受信。初期値配信を停止。")
        self._publish(list(msg.name), list(msg.position))

    def _publish(self, names, positions):
        fb = JointState()
        fb.header.stamp = self.get_clock().now().to_msg()
        fb.name = names
        fb.position = positions
        self._pub.publish(fb)


def main(args=None):
    rclpy.init(args=args)
    node = MockServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
