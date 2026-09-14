import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 找到相關的 package 路徑
    gazebo_sim = get_package_share_directory('gazebo_sim')
    auo_model_share = get_package_share_directory('auo_model')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # 設定 Gazebo 模型路徑，確保能找到 Target 模型
    model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=[
            PathJoinSubstitution([auo_model_share, '..']),
            ':',
            EnvironmentVariable('GAZEBO_MODEL_PATH', default_value=''),
        ]
    )
    
    world_path = os.path.join(gazebo_sim, 'worlds', 'school_8F.world')

    # --- 3. 讀取啟動參數 ---
    robot_model = LaunchConfiguration('robot_model', default='AMR150M')
    init_x = LaunchConfiguration('init_x', default='1.0')
    init_y = LaunchConfiguration('init_y', default='1.0')
    init_a = LaunchConfiguration('init_a', default='0.0')
    
    # 新增：目標點座標參數
    target_x = LaunchConfiguration('target_x', default='0.0')
    target_y = LaunchConfiguration('target_y', default='0.0')

    debug = LaunchConfiguration('debug', default='false')
    gui = LaunchConfiguration('gui', default='true')
    paused = LaunchConfiguration('paused', default='false')
    headless = LaunchConfiguration('headless', default='false')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # --- 節點設定 ---
    
    # 1. 麥克納姆輪控制器
    mecanum_controller_params = PathJoinSubstitution([
        gazebo_sim, 'config', robot_model, 'amr_config.yaml',
    ])
    
    mecanum_controller_node = Node(
        package='gazebo_sim',
        executable='mecanum_controller.py',
        name='mecanum_controller',
        output='screen',
        parameters=[mecanum_controller_params, {'use_sim_time': use_sim_time}],
    )

    # 2. ROS 2 控制器 Spawner
    controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_controller', 'front_right_controller', 'front_left_controller', 'rear_right_controller', 'rear_left_controller'],
    )

    # 3. Gazebo 模擬器
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            'world': world_path, 'debug': debug, 'gui': gui,
            'paused': paused, 'headless': headless, 'verbose': 'true',
        }.items(),
    )
    
    # 4. 機器人狀態與生成
    robot_state_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(auo_model_share, 'launch', 'robot_state.launch.py')),
        launch_arguments={
            'robot_model': robot_model, 'init_x': init_x, 'init_y': init_y,
            'init_a': init_a, 'use_sim_time': use_sim_time,
        }.items(),
    )

    # 5. Laser Merger
    laserscan_merger_node = Node(
        package='ira_laser_tools',
        executable='laserscan_multi_merger',
        name='laserscan_merger',
        parameters=[{
            'destination_frame': 'base_link',
            'scan_destination_topic': 'scan',
            'laserscan_topics': '/urg_front/scan /urg_back/scan',
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )

    # ==================================================================
    # == 新增：生成目標點 (Target) 節點
    # ==================================================================
    # 假設你的 Target 模型放在 auo_model/models/Target/model.sdf
    target_sdf_path = os.path.join(auo_model_share, 'urdf', 'Target', 'model.sdf')

    spawn_target_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='target_spawner',
        arguments=[
            '-entity', 'Target',      # Gazebo 中的名稱
            '-file', target_sdf_path,      # 模型檔案路徑
            '-x', target_x,                # X 座標
            '-y', target_y,                # Y 座標
            '-z', '0.05'                   # 稍微離地，避免物理碰撞重疊
        ],
        output='screen'
    )
    
    # --- 5. 組合 ---
    return LaunchDescription([
        model_path,
        mecanum_controller_node,
        controller_spawner,
        gazebo_launch,
        robot_state_launch,
        laserscan_merger_node,
        spawn_target_node,  # <--- 加入這一行
    ])