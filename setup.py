from setuptools import setup, find_packages
import os
from glob import glob


package_name = 'three_d_mouse_leader'


def _collect_data_files(base_dir: str):
    """base_dir 配下の全ファイルを ament data_files 形式で返す。"""
    data_files = []
    for root, _, files in os.walk(base_dir):
        if not files:
            continue
        rel_root = os.path.relpath(root, '.')
        install_dir = os.path.join('share', package_name, rel_root)
        src_files = [os.path.join(root, f) for f in files]
        data_files.append((install_dir, src_files))
    return data_files


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
    ] + _collect_data_files('urdf') + _collect_data_files('meshes'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='inoma',
    maintainer_email='inoma@users.noreply.github.com',
    description='SpaceMouse to SO-ARM101 differential IK ROS2 node using frax',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'spacemouse_ik_node = three_d_mouse_leader.spacemouse_ik_node:main',
            'mock_servo_node = three_d_mouse_leader.mock_servo_node:main',
        ],
    },
)
