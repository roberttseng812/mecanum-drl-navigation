import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    # --- 1. 宣告啟動參數 ---
    pkg_gazebo_sim = FindPackageShare('gazebo_sim')
    # 這些參數是從 simulation.launch.py 傳遞進來的
    declare_robot_model_arg = DeclareLaunchArgument(
        'robot_model',
        default_value='AMR150M',
        description='Name of the robot model to use (e.g., AMR150M)'
    )
    declare_init_x_arg = DeclareLaunchArgument(
        'init_x', 
        default_value='0.0',
        description='Initial X position of the robot'
    )
    declare_init_y_arg = DeclareLaunchArgument(
        'init_y', 
        default_value='0.0',
        description='Initial Y position of the robot'
    )
    declare_init_a_arg = DeclareLaunchArgument(
        'init_a', 
        default_value='0.0',
        description='Initial Yaw angle of the robot'
    )
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock'
    )
    
    # 這是 robot_state.launch 自己的參數
    # *** 注意：已將預設值從 '/' (ROS 1) 更改為 'amr' (ROS 2) ***
    declare_robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='amr', 
        description='Entity name for the robot in Gazebo'
    )
    gazebo_controller_config_path = PathJoinSubstitution([
    pkg_gazebo_sim,
    'config',
    'gazebo_controller.yaml'
])
  

    # --- 2. 獲取啟動配置 ---
    
    robot_model = LaunchConfiguration('robot_model')
    init_x = LaunchConfiguration('init_x')
    init_y = LaunchConfiguration('init_y')
    init_a = LaunchConfiguration('init_a')
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    
    # --- 3. 定義機器人描述 (Robot Description) ---
    
    # 獲取 XACRO 檔案的路徑
    xacro_file_path = PathJoinSubstitution([
        FindPackageShare('auo_model'),
        'robot', # 根據原始檔案的路徑
        [robot_model, '.xacro']
    ])
    
    # 處理 XACRO 檔案以獲取 robot_description
    # 我們使用 Command 替換項在啟動時執行 xacro
    robot_description_content = Command([
        'xacro ', xacro_file_path,
        ' gazebo_controllers:=', gazebo_controller_config_path
    ])

    robot_description_param = {'robot_description': robot_description_content}
    
    # --- 4. 定義節點 (Nodes) ---
    
    # Robot State Publisher
    # 這個節點會根據 /joint_states 和 robot_description 來發布 TF
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            robot_description_param,
            {'use_sim_time': use_sim_time},
            # 轉換 <param name="tf_prefix" value="$(arg robot_name)"/>
            #{'frame_prefix': robot_name} 
        ]
    )
    
    # Joint State Publisher (已移除)
    # ROS 1 中的 <node name="joint_state_publisher" ... />
    # 在使用 ros2_control 時**不需要**。
    # (在 simulation.launch.py 中啟動的) 'joint_state_controller'
    # 已經從 Gazebo 發布 /joint_states。
    # 包含此節點會導致衝突。

    # Spawn Entity (在 Gazebo 中生成模型)
    # 這個節點會將機器人模型生成到 Gazebo 中
    # 它取代了 <node pkg="gazebo_ros" type="spawn_model" ... />
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_entity',
        output='screen',
        arguments=[
            '-topic', 'robot_description', # 訂閱 robot_state_publisher 的主題
            '-entity', robot_name,        # 在 Gazebo 中的模型名稱
            '-x', init_x,
            '-y', init_y,
            '-Y', init_a,                 # -Y 是 Yaw (航向角)
        ]
    )
    
    # --- 5. 建立 LaunchDescription ---
    
    ld = LaunchDescription()
    
    # 添加啟動參數
    ld.add_action(declare_robot_model_arg)
    ld.add_action(declare_init_x_arg)
    ld.add_action(declare_init_y_arg)
    ld.add_action(declare_init_a_arg)
    ld.add_action(declare_use_sim_time_arg)
    ld.add_action(declare_robot_name_arg)
    
    # 添加節點
    ld.add_action(robot_state_publisher_node)
    ld.add_action(spawn_entity_node)
    
    return ld
