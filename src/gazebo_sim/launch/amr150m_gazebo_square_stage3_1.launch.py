import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    gazebo_sim = get_package_share_directory('gazebo_sim')
    auo_model_share = get_package_share_directory('auo_model')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    realsence_share = get_package_share_directory('realsense2_description')

    model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=[
            PathJoinSubstitution([auo_model_share, '..']),
            ':',
            PathJoinSubstitution([realsence_share, '..']),
            ':',
            EnvironmentVariable('GAZEBO_MODEL_PATH', default_value=''),
        ]
    )

    nvidia_offload_1 = SetEnvironmentVariable(
        name='__NV_PRIME_RENDER_OFFLOAD',
        value='1'
    )
    nvidia_offload_2 = SetEnvironmentVariable(
        name='__GLX_VENDOR_LIBRARY_NAME',
        value='nvidia'
    )

    world_path = os.path.join(gazebo_sim, 'worlds', 'school_8f_square.world')

    robot_model = LaunchConfiguration('robot_model', default='AMR150M')
    init_x = LaunchConfiguration('init_x', default='1.0')
    init_y = LaunchConfiguration('init_y', default='1.0')
    init_a = LaunchConfiguration('init_a', default='0.0')
    target_x = LaunchConfiguration('target_x', default='0.0')
    target_y = LaunchConfiguration('target_y', default='0.0')
    debug = LaunchConfiguration('debug', default='false')
    gui = LaunchConfiguration('gui', default='true')
    paused = LaunchConfiguration('paused', default='false')
    headless = LaunchConfiguration('headless', default='false')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

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

    controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_controller',
            'front_right_controller',
            'front_left_controller',
            'rear_right_controller',
            'rear_left_controller'
        ],
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            'world': world_path,
            'debug': debug,
            'gui': gui,
            'paused': paused,
            'headless': headless,
            'verbose': 'true',
        }.items(),
    )

    robot_state_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(auo_model_share, 'launch', 'robot_state.launch.py')),
        launch_arguments={
            'robot_model': robot_model,
            'init_x': init_x,
            'init_y': init_y,
            'init_a': init_a,
            'use_sim_time': use_sim_time,
        }.items(),
    )

    laserscan_merger_node = Node(
        package='ira_laser_tools',
        executable='laserscan_multi_merger',
        name='laserscan_merger',
        parameters=[{
            'destination_frame': 'base_link',
            'scan_destination_topic': 'scan',
            'laserscan_topics': '/urg_front/scan /urg_back/scan',
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    target_sdf_path = os.path.join(auo_model_share, 'urdf', 'Target', 'model.sdf')
    obstacle_sdf_path = os.path.join(auo_model_share, 'urdf', 'ObstacleBox', 'model.sdf')

    spawn_target_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='target_spawner',
        arguments=[
            '-entity', 'Target',
            '-file', target_sdf_path,
            '-x', target_x,
            '-y', target_y,
            '-z', '0.05',
        ],
        output='screen'
    )

    obstacle_spawners = []
    for i in range(3):
        obstacle_spawners.append(
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                name=f'obstacle_spawner_{i}',
                arguments=[
                    '-entity', f'Obstacle_{i}',
                    '-file', obstacle_sdf_path,
                    '-x', '50.0',
                    '-y', '50.0',
                    '-z', '0.0',
                ],
                output='screen'
            )
        )

    spawn_entities_delayed = TimerAction(
        period=2.0,
        actions=[spawn_target_node, *obstacle_spawners],
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_model', default_value='AMR150M'),
        DeclareLaunchArgument('init_x', default_value='1.0'),
        DeclareLaunchArgument('init_y', default_value='1.0'),
        DeclareLaunchArgument('init_a', default_value='0.0'),
        DeclareLaunchArgument('target_x', default_value='0.0'),
        DeclareLaunchArgument('target_y', default_value='0.0'),
        DeclareLaunchArgument('debug', default_value='false'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('paused', default_value='false'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        model_path,
        nvidia_offload_1,
        nvidia_offload_2,
        mecanum_controller_node,
        controller_spawner,
        gazebo_launch,
        robot_state_launch,
        laserscan_merger_node,
        spawn_entities_delayed,
    ])
