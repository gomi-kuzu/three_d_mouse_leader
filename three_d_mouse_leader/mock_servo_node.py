#!/usr/bin/env python3
"""
Mock Servo Node (デバッグ用仮想サーボ)

起動時に初期関節角を /lekiwi/joint_states に配信し、
その後は /lekiwi/arm_joint_commands を受け取って指令値を
そのまま現在値として返す。

実機なしで IK ノードと Rviz の動作確認に使用する。

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
    _DEFAULT_CMD_JOINT_NAMES = (
        "arm_shoulder_pan",
        "arm_shoulder_lift",
        "arm_elbow_flex",
        "arm_wrist_flex",
        "arm_wrist_roll",
        "arm_gripper",
    )
    _DEFAULT_INIT_DEG = (0.0, -45.0, 90.0, -45.0, 0.0, 0.0)

    def __init__(self):
        super().__init__("mock_servo_node")

        self.declare_parameter(
            "init_joint_positions",
            ",".join(str(d) for d in self._DEFAULT_INIT_DEG),
        )
        self.declare_parameter(
            "joint_names_cmd",
            ",".join(self._DEFAULT_CMD_JOINT_NAMES),
        )

        init_deg_str: str = self.get_parameter("init_joint_positions").value
        self._init_positions = [
            math.radians(float(v.strip())) for v in init_deg_str.split(",")
        ]

        names_str: str = self.get_parameter("joint_names_cmd").value
        self._joint_names = [n.strip() for n in names_str.split(",")]

        # 不足分を 0 で補完
        while len(self._init_positions) < len(self._joint_names):
            self._init_positions.append(0.0)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub = self.create_publisher(JointState, "/lekiwi/joint_states", qos)
        self._sub = self.create_subscription(
            JointState,
            "/lekiwi/arm_joint_commands",
            self._cmd_callback,
            qos,
        )

        # コマンド受信前は初期値を定期配信し続ける (IKノードの接続待ちに対応)
        self._received_cmd = False
        self._init_timer = self.create_timer(0.1, self._publish_initial)

        self.get_logger().info(
            f"MockServoNode 起動: 初期角 {[round(math.degrees(p), 1) for p in self._init_positions]} [deg]"
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
