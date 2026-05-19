#!/usr/bin/env python3
"""
3d_mouse_leader パッケージの起動ファイル

使用方法:
    ros2 launch three_d_mouse_leader spacemouse_ik.launch.py \
        urdf_path:=/path/to/so101_new_calib.urdf

オプション引数 (すべて launch 引数で上書き可能):
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
from launch.substitutions import LaunchConfiguration, Command
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
            default_value='0.0,-78.0,82.0,62.0,0.0', #lekiwi home position
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
            default_value='false',
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
            default_value='false',
            description='SpaceMouse 入力方向矢印を Rviz に表示するか否か',
        ),
        DeclareLaunchArgument(
            'velocity_frame',
            default_value='world',
            description='手先速度指令の基準座標系: "world" = base_link 系, "ee" = 手先座標系',
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
    default_rviz_config = os.path.join(pkg_dir, 'rviz', 'spacemouse_ik.rviz')

    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', default_rviz_config],
        condition=launch.conditions.IfCondition(LaunchConfiguration('use_rviz')),
    )

    mock_servo_node = launch_ros.actions.Node(
        package='three_d_mouse_leader',
        executable='mock_servo_node',
        name='mock_servo_node',
        output='screen',
        parameters=[{
            'init_joint_positions': LaunchConfiguration('init_joint_positions'),
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
            'use_gripper_tip_ee':  LaunchConfiguration('use_gripper_tip_ee'),
            'joint_names_so101':   LaunchConfiguration('joint_names_so101'),
            'enable_trail':        LaunchConfiguration('enable_trail'),
            'enable_ee_sphere':    LaunchConfiguration('enable_ee_sphere'),
            'enable_ee_axes':      LaunchConfiguration('enable_ee_axes'),
            'enable_input_arrows': LaunchConfiguration('enable_input_arrows'),
            'velocity_frame':      LaunchConfiguration('velocity_frame'),
        }],
    )

    return launch.LaunchDescription(args + [
        robot_state_publisher_node,
        spacemouse_ik_node,
        mock_servo_node,
        rviz_node,
    ])
