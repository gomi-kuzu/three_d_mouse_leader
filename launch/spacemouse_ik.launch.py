#!/usr/bin/env python3
"""
3d_mouse_leader パッケージの起動ファイル

使用方法:
    ros2 launch three_d_mouse_leader spacemouse_ik.launch.py \
        urdf_path:=/path/to/so101_new_calib.urdf

オプション引数 (すべて launch 引数で上書き可能):
    robot_type             : ロボット種類 "so101" / "mycobot280" (デフォルト "so101")
    urdf_path              : URDF ファイルパス (必須)
    control_frequency      : 制御周波数 [Hz] (デフォルト 30.0)
    lin_gain_x/y/z         : 並進ゲイン (デフォルト 0.10)
    rot_gain_roll/pitch/yaw: 回転ゲイン (デフォルト 0.30)
    deadzone               : 不感帯 (デフォルト 0.02)
    init_joint_positions   : 初期関節角 degree, カンマ区切り (例: "0,-45,90,-45,0")
    device_path            : SpaceMouse デバイスパス (空 = 自動検出)
    gripper_init_deg       : グリッパー初期角 [degree] (デフォルト 0.0)
    velocity_frame         : 手先速度指令の基準座標系 "world" / "ee" (デフォルト "world")
"""

import launch
import launch_ros.actions
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch.actions import DeclareLaunchArgument
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('three_d_mouse_leader')
    default_urdf = os.path.join(pkg_dir, 'urdf', 'so101_new_calib.urdf')

    # ── Launch 引数定義 ────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            'robot_type',
            default_value='so101',
            description='ロボット種類: "so101" / "mycobot280"',
        ),
        DeclareLaunchArgument(
            'urdf_path',
            default_value=default_urdf,
            description='SO-ARM101 の URDF ファイルパス',
        ),
        DeclareLaunchArgument(
            'control_frequency',
            default_value='30.0',
            description='制御ループ周波数 [Hz]',
        ),
        DeclareLaunchArgument(
            'lin_gain_x',
            default_value='0.10',
            description='並進 X 軸ゲイン [m/s per unit]',
        ),
        DeclareLaunchArgument(
            'lin_gain_y',
            default_value='0.10',
            description='並進 Y 軸ゲイン [m/s per unit]',
        ),
        DeclareLaunchArgument(
            'lin_gain_z',
            default_value='0.10',
            description='並進 Z 軸ゲイン [m/s per unit]',
        ),
        DeclareLaunchArgument(
            'rot_gain_roll',
            default_value='0.30',
            description='回転 Roll ゲイン [rad/s per unit]',
        ),
        DeclareLaunchArgument(
            'rot_gain_pitch',
            default_value='0.30',
            description='回転 Pitch ゲイン [rad/s per unit]',
        ),
        DeclareLaunchArgument(
            'rot_gain_yaw',
            default_value='0.30',
            description='回転 Yaw ゲイン [rad/s per unit]',
        ),
        DeclareLaunchArgument(
            'deadzone',
            default_value='0.02',
            description='SpaceMouse 入力の不感帯',
        ),
        DeclareLaunchArgument(
            'init_joint_positions',
            # default_value='0,-45,90,-45,0',
            default_value='0.0,50.0,-130.0,30.0,0.0,0.0', #myCobot
            # default_value='0.0,-78.0,82.0,62.0,0.0', #lekiwi home position
            description='初期関節角 [degree], カンマ区切り (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)',
        ),
        DeclareLaunchArgument(
            'device_path',
            default_value='',
            description='SpaceMouse デバイスパス (空 = 自動検出)',
        ),
        DeclareLaunchArgument(
            'gripper_init_deg',
            default_value='0.0',
            description='グリッパー初期角 [degree]',
        ),
        DeclareLaunchArgument(
            'gripper_min_deg',
            default_value='-10.0',
            description='グリッパー最小角 [degree] (auto_gripper_limits_from_urdf=true なら上書き)',
        ),
        DeclareLaunchArgument(
            'gripper_max_deg',
            default_value='100.0',
            description='グリッパー最大角 [degree] (auto_gripper_limits_from_urdf=true なら上書き)',
        ),
        DeclareLaunchArgument(
            'gripper_speed_dps',
            default_value='40.0',
            description='グリッパー開閉速度 [degree/s]',
        ),
        DeclareLaunchArgument(
            'auto_gripper_limits_from_urdf',
            default_value='true',
            description='mycobot時にURDFのgripper_controller制限で min/max を自動設定',
        ),
        DeclareLaunchArgument(
            'publish_mycobot_gripper_value',
            default_value='true',
            description='mycobot用の実機グリッパー値トピックを publish するか',
        ),
        DeclareLaunchArgument(
            'mycobot_gripper_value_topic',
            default_value='/mycobot/gripper/value',
            description='mycobotグリッパー開度値トピック (std_msgs/Int32)',
        ),
        DeclareLaunchArgument(
            'mycobot_gripper_value_min',
            default_value='0',
            description='mycobotグリッパー開度最小値',
        ),
        DeclareLaunchArgument(
            'mycobot_gripper_value_max',
            default_value='100',
            description='mycobotグリッパー開度最大値',
        ),
        DeclareLaunchArgument(
            'use_gripper_tip_ee',
            default_value='true',
            description='IK の EE をグリッパ先端 (true) / 手首付け根 (false) に設定',
        ),
        DeclareLaunchArgument(
            'joint_names_so101',
            default_value='shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll',
            description='制御対象の URDF 関節名 (カンマ区切り)',
        ),
        DeclareLaunchArgument(
            'joint_names_mycobot',
            default_value='joint2_to_joint1,joint3_to_joint2,joint4_to_joint3,joint5_to_joint4,joint6_to_joint5,joint6output_to_joint6',
            description='myCobot 用の制御対象 URDF 関節名 (frax が使用する joint 名, カンマ区切り)',
        ),
        DeclareLaunchArgument(
            'joint_state_topic',
            default_value='',
            description='現在関節角トピック (空文字なら robot_type 既定値を使用)',
        ),
        DeclareLaunchArgument(
            'joint_command_topic',
            default_value='',
            description='関節指令トピック (空文字なら robot_type 既定値を使用)',
        ),
        DeclareLaunchArgument(
            'feedback_joint_names',
            default_value='',
            description='フィードバック JointState の関節名 (空文字なら robot_type 既定値を使用)',
        ),
        DeclareLaunchArgument(
            'command_joint_names',
            default_value='',
            description='コマンド JointState の関節名 (空文字なら robot_type 既定値を使用)',
        ),
        DeclareLaunchArgument(
            'feedback_unit_mode',
            default_value='auto',
            description='フィードバック単位: auto / range_m100_100 / rad / deg',
        ),
        DeclareLaunchArgument(
            'command_unit_mode',
            default_value='auto',
            description='コマンド単位: auto / range_m100_100 / rad / deg',
        ),
        DeclareLaunchArgument(
            'base_frame_id',
            default_value='',
            description='可視化/入力矢印の基準フレーム (空文字なら robot_type 既定値)',
        ),
        DeclareLaunchArgument(
            'ee_frame_id',
            default_value='',
            description='EE マーカー/EE入力矢印フレーム (空文字なら robot_type 既定値)',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Rviz2 を起動するか否か',
        ),
        DeclareLaunchArgument(
            'use_mock_servo',
            default_value='false',
            description='仮想サーボノードを起動する (実機なしデバッグ用)',
        ),
        DeclareLaunchArgument(
            'enable_trail',
            default_value='false', #TODO:描写が重くなるバグなどを修正
            description='手先軌跡を Rviz に表示するか否か',
        ),
        DeclareLaunchArgument(
            'enable_ee_sphere',
            default_value='true',
            description='手先球マーカーを Rviz に表示するか否か',
        ),
        DeclareLaunchArgument(
            'enable_ee_axes',
            default_value='true',
            description='手先姿勢を RGB 軸矢印で Rviz に表示するか否か',
        ),
        DeclareLaunchArgument(
            'enable_input_arrows',
            default_value='true',
            description='SpaceMouse 入力方向矢印を Rviz に表示するか否か',
        ),
        DeclareLaunchArgument(
            'velocity_frame',
            default_value='world',
            description='手先速度指令の基準座標系: "world" = base_link 系, "ee" = 手先座標系',
        ),
        DeclareLaunchArgument(
            'singularity_avoidance',
            default_value='true',
            description='特異点回避を有効にする (DLS 擬似逆行列を使用)',
        ),
        DeclareLaunchArgument(
            'variable_damping',
            default_value='true',
            description='可操作性に応じた可変ダンピングを使用 (false = 固定ダンピング)',
        ),
        DeclareLaunchArgument(
            'damping_lambda',
            default_value='0.05',
            description='DLS ダンピング係数 λ (大きいほど特異点で安定、追従性低下)',
        ),
        DeclareLaunchArgument(
            'manipulability_threshold',
            default_value='0.04',
            description='可変ダンピング開始閾値 w0 (可操作性がこれ以下になると λ が増大)',
        ),
        DeclareLaunchArgument(
            'feedback_correction_enabled',
            default_value='false',
            description='実測角への補正を有効化 (既定: false)',
        ),
        DeclareLaunchArgument(
            'feedback_correction_only_when_commanding',
            default_value='true',
            description='操作入力があるときだけ補正を適用 (ドリフト再発防止)',
        ),
        DeclareLaunchArgument(
            'feedback_correction_alpha',
            default_value='0.2',
            description='実測角への補正ゲイン (0.0-1.0)',
        ),
        DeclareLaunchArgument(
            'feedback_correction_deadband_deg',
            default_value='0.8',
            description='この角度差未満は補正しないデッドバンド [deg]',
        ),
        DeclareLaunchArgument(
            'feedback_correction_max_step_deg',
            default_value='0.4',
            description='1周期あたりの最大補正量 [deg]',
        ),
        DeclareLaunchArgument(
            'feedback_correction_command_threshold',
            default_value='0.01',
            description='操作あり判定の task_vel ノルム閾値',
        ),
        DeclareLaunchArgument(
            'profile_ik_timing',
            default_value='false',
            description='IK 計算時間の計測ログを有効化',
        ),
        DeclareLaunchArgument(
            'timing_log_every_n',
            default_value='120',
            description='IK 計測ログを出すサンプル間隔',
        ),
    ]

    # ── robot_state_publisher ───────────────────────────────────────────────
    robot_description_content = ParameterValue(
        Command(['cat ', LaunchConfiguration('urdf_path')]),
        value_type=str,
    )

    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'publish_frequency': 30.0,
        }],
    )

    # ── Rviz2 ──────────────────────────────────────────────────────────────
    default_rviz_config_so101 = os.path.join(pkg_dir, 'rviz', 'spacemouse_ik.rviz')
    default_rviz_config_mycobot = os.path.join(pkg_dir, 'rviz', 'spacemouse_ik_mycobot.rviz')

    rviz_node_so101 = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', default_rviz_config_so101],
        condition=launch.conditions.IfCondition(
            PythonExpression([
                '"', LaunchConfiguration('use_rviz'), '" == "true" and '
                '"', LaunchConfiguration('robot_type'), '" not in ["mycobot", "mycobot280", "mycobot280_m5"]'
            ])
        ),
    )

    rviz_node_mycobot = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', default_rviz_config_mycobot],
        condition=launch.conditions.IfCondition(
            PythonExpression([
                '"', LaunchConfiguration('use_rviz'), '" == "true" and '
                '"', LaunchConfiguration('robot_type'), '" in ["mycobot", "mycobot280", "mycobot280_m5"]'
            ])
        ),
    )

    mock_servo_node = launch_ros.actions.Node(
        package='three_d_mouse_leader',
        executable='mock_servo_node',
        name='mock_servo_node',
        output='screen',
        parameters=[{
            'robot_type': LaunchConfiguration('robot_type'),
            'init_joint_positions': LaunchConfiguration('init_joint_positions'),
            'joint_state_topic': LaunchConfiguration('joint_state_topic'),
            'joint_command_topic': LaunchConfiguration('joint_command_topic'),
            'feedback_unit_mode': LaunchConfiguration('feedback_unit_mode'),
            'joint_names_cmd': LaunchConfiguration('command_joint_names'),
        }],
        condition=launch.conditions.IfCondition(LaunchConfiguration('use_mock_servo')),
    )

    # ── ノード定義 ─────────────────────────────────────────────────────────────
    spacemouse_ik_node = launch_ros.actions.Node(
        package='three_d_mouse_leader',
        executable='spacemouse_ik_node',
        name='spacemouse_ik_node',
        output='screen',
        parameters=[{
            'robot_type':          LaunchConfiguration('robot_type'),
            'urdf_path':           LaunchConfiguration('urdf_path'),
            'control_frequency':   LaunchConfiguration('control_frequency'),
            'lin_gain_x':          LaunchConfiguration('lin_gain_x'),
            'lin_gain_y':          LaunchConfiguration('lin_gain_y'),
            'lin_gain_z':          LaunchConfiguration('lin_gain_z'),
            'rot_gain_roll':       LaunchConfiguration('rot_gain_roll'),
            'rot_gain_pitch':      LaunchConfiguration('rot_gain_pitch'),
            'rot_gain_yaw':        LaunchConfiguration('rot_gain_yaw'),
            'deadzone':            LaunchConfiguration('deadzone'),
            'init_joint_positions': LaunchConfiguration('init_joint_positions'),
            'device_path':         LaunchConfiguration('device_path'),
            'gripper_init_deg':    LaunchConfiguration('gripper_init_deg'),
            'gripper_min_deg':     LaunchConfiguration('gripper_min_deg'),
            'gripper_max_deg':     LaunchConfiguration('gripper_max_deg'),
            'gripper_speed_dps':   LaunchConfiguration('gripper_speed_dps'),
            'auto_gripper_limits_from_urdf': LaunchConfiguration('auto_gripper_limits_from_urdf'),
            'publish_mycobot_gripper_value': LaunchConfiguration('publish_mycobot_gripper_value'),
            'mycobot_gripper_value_topic': LaunchConfiguration('mycobot_gripper_value_topic'),
            'mycobot_gripper_value_min': LaunchConfiguration('mycobot_gripper_value_min'),
            'mycobot_gripper_value_max': LaunchConfiguration('mycobot_gripper_value_max'),
            'use_gripper_tip_ee':  LaunchConfiguration('use_gripper_tip_ee'),
            'joint_names_so101':   LaunchConfiguration('joint_names_so101'),
            'joint_names_mycobot': LaunchConfiguration('joint_names_mycobot'),
            'joint_state_topic':   LaunchConfiguration('joint_state_topic'),
            'joint_command_topic': LaunchConfiguration('joint_command_topic'),
            'feedback_joint_names': LaunchConfiguration('feedback_joint_names'),
            'command_joint_names': LaunchConfiguration('command_joint_names'),
            'feedback_unit_mode':  LaunchConfiguration('feedback_unit_mode'),
            'command_unit_mode':   LaunchConfiguration('command_unit_mode'),
            'base_frame_id':       LaunchConfiguration('base_frame_id'),
            'ee_frame_id':         LaunchConfiguration('ee_frame_id'),
            'enable_trail':        LaunchConfiguration('enable_trail'),
            'enable_ee_sphere':    LaunchConfiguration('enable_ee_sphere'),
            'enable_ee_axes':      LaunchConfiguration('enable_ee_axes'),
            'enable_input_arrows': LaunchConfiguration('enable_input_arrows'),
            'velocity_frame':      LaunchConfiguration('velocity_frame'),
            'singularity_avoidance': LaunchConfiguration('singularity_avoidance'),
            'variable_damping':    LaunchConfiguration('variable_damping'),
            'damping_lambda':      LaunchConfiguration('damping_lambda'),
            'manipulability_threshold': LaunchConfiguration('manipulability_threshold'),
            'feedback_correction_enabled': LaunchConfiguration('feedback_correction_enabled'),
            'feedback_correction_only_when_commanding': LaunchConfiguration('feedback_correction_only_when_commanding'),
            'feedback_correction_alpha': LaunchConfiguration('feedback_correction_alpha'),
            'feedback_correction_deadband_deg': LaunchConfiguration('feedback_correction_deadband_deg'),
            'feedback_correction_max_step_deg': LaunchConfiguration('feedback_correction_max_step_deg'),
            'feedback_correction_command_threshold': LaunchConfiguration('feedback_correction_command_threshold'),
            'profile_ik_timing': LaunchConfiguration('profile_ik_timing'),
            'timing_log_every_n': LaunchConfiguration('timing_log_every_n'),
        }],
    )

    return launch.LaunchDescription(args + [
        robot_state_publisher_node,
        spacemouse_ik_node,
        mock_servo_node,
        rviz_node_so101,
        rviz_node_mycobot,
    ])
