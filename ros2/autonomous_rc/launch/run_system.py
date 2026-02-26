"""
run_system.py — 단일 노드 실행 런치 파일
  엔진 경로 등 파라미터 주입

사용법:
  ros2 launch autonomous_rc run_system.py
  ros2 launch autonomous_rc run_system.py base_dir:=/path/to/models video:=test.mp4
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ─── 파라미터 선언 ───
        DeclareLaunchArgument(
            'base_dir',
            default_value='/home/user/Downloads',
            description='엔진 파일 / 영상 베이스 디렉토리'
        ),
        DeclareLaunchArgument(
            'video',
            default_value='starting.mp4',
            description='입력 영상 파일명'
        ),
        DeclareLaunchArgument(
            'lane_engine',
            default_value='seg_regnety_fp16.engine',
            description='Lane segmentation TRT 엔진 파일명'
        ),
        DeclareLaunchArgument(
            'od_engine',
            default_value='OD.engine',
            description='Object Detection TRT 엔진 파일명'
        ),
        DeclareLaunchArgument(
            'classes_txt',
            default_value='classes.txt',
            description='클래스 이름 텍스트 파일명'
        ),

        # ─── 노드 실행 ───
        Node(
            package='autonomous_rc',
            executable='autonomous_rc_node',
            name='autonomous_rc_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'base_dir':     LaunchConfiguration('base_dir'),
                'video':        LaunchConfiguration('video'),
                'lane_engine':  LaunchConfiguration('lane_engine'),
                'od_engine':    LaunchConfiguration('od_engine'),
                'classes_txt':  LaunchConfiguration('classes_txt'),
            }]
        ),
    ])
