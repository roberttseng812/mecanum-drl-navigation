import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_amr_nav2_drl = get_package_share_directory('amr_nav2_drl')
    pkg_gazebo_sim = get_package_share_directory('gazebo_sim')
    pkg_auo_model = get_package_share_directory('auo_model')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    realsense_share = get_package_share_directory('realsense2_description')


    model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=[
            PathJoinSubstitution([pkg_auo_model, '..']),
            ':',
            PathJoinSubstitution([realsense_share, '..']),
            ':',
            EnvironmentVariable('GAZEBO_MODEL_PATH', default_value=''),
        ]
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_param = {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}

    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    model_file = LaunchConfiguration('model')
    drl_params_file = LaunchConfiguration('drl_params_file')

    debug = LaunchConfiguration('debug')
    gui = LaunchConfiguration('gui')
    paused = LaunchConfiguration('paused')
    headless = LaunchConfiguration('headless')

    init_x = LaunchConfiguration('init_x')
    init_y = LaunchConfiguration('init_y')
    init_a = LaunchConfiguration('init_a')

    robot_model = LaunchConfiguration('robot_model')
    use_laser_merger = LaunchConfiguration('use_laser_merger')

    world_path = os.path.join(pkg_gazebo_sim, 'worlds', 'school_8f_square.world')
    default_map = os.path.join(pkg_gazebo_sim, 'maps', 'school_8F_square.yaml')
    default_params = os.path.join(pkg_amr_nav2_drl, 'config', 'nav2_params_drl_stage3.yaml')
    default_drl_params = os.path.join(pkg_amr_nav2_drl, 'config', 'drl_inference_stage3.yaml')
    rviz_config = os.path.join(pkg_nav2_bringup, 'rviz', 'nav2_default_view.rviz')

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='true'))
    ld.add_action(DeclareLaunchArgument('map', default_value=default_map))
    ld.add_action(DeclareLaunchArgument('params_file', default_value=default_params))
    ld.add_action(DeclareLaunchArgument('drl_params_file', default_value=default_drl_params))
    ld.add_action(DeclareLaunchArgument('model', default_value='models_stage3/sac_stage3_interrupt.zip'))#models_stage3/sac_stage3_interrupt.zip

    ld.add_action(DeclareLaunchArgument('debug', default_value='false'))
    ld.add_action(DeclareLaunchArgument('gui', default_value='true'))
    ld.add_action(DeclareLaunchArgument('paused', default_value='false'))
    ld.add_action(DeclareLaunchArgument('headless', default_value='false'))

    ld.add_action(DeclareLaunchArgument('init_x', default_value='-4.0'))
    ld.add_action(DeclareLaunchArgument('init_y', default_value='0.5'))
    ld.add_action(DeclareLaunchArgument('init_a', default_value='0.0'))

    ld.add_action(DeclareLaunchArgument('robot_model', default_value='AMR150M'))
    ld.add_action(DeclareLaunchArgument('use_laser_merger', default_value='true'))

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            'world': world_path,
            'debug': debug,
            'gui': gui,
            'paused': paused,
            'headless': headless,
            'verbose': 'false',
        }.items(),
    )

    robot_state_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_auo_model, 'launch', 'robot_state.launch.py')),
        launch_arguments={
            'robot_model': robot_model,
            'init_x': init_x,
            'init_y': init_y,
            'init_a': init_a,
            'use_sim_time': use_sim_time,
            'robot_name': 'amr',
        }.items(),
    )

    controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_controller',
            'front_right_controller',
            'front_left_controller',
            'rear_right_controller',
            'rear_left_controller',
        ],
        parameters=[use_sim_time_param],
    )

    mecanum_controller_params = PathJoinSubstitution([
        pkg_gazebo_sim, 'config', robot_model, 'amr_config.yaml',
    ])
    mecanum_controller_node = Node(
        package='gazebo_sim',
        executable='mecanum_controller.py',
        name='mecanum_controller',
        parameters=[mecanum_controller_params, use_sim_time_param],
    )

    laser_merger_node = Node(
        package='ira_laser_tools',
        executable='laserscan_multi_merger',
        name='laserscan_merger',
        parameters=[
            {
                'destination_frame': 'base_link',
                'scan_destination_topic': 'scan',
                'laserscan_topics': '/urg_front/scan /urg_back/scan',
            },
            use_sim_time_param,
        ],
        condition=IfCondition(use_laser_merger),
    )

    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_nav2_bringup, 'launch', 'bringup_launch.py')),
        launch_arguments={
            'slam': 'False',
            'map': map_yaml,
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': 'true',
            'use_composition': 'False',
        }.items(),
    )

    drl_inference_node = Node(
        package='amr_nav2_drl',
        executable='drl_inference_node_stage3.py',
        name='drl_inference_stage3',
        parameters=[
            use_sim_time_param,
            drl_params_file,
            {'model_path': model_file},
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[use_sim_time_param],
    )

    ld.add_action(model_path)
    ld.add_action(gazebo_launch)
    ld.add_action(robot_state_launch)
    ld.add_action(controller_spawner)
    ld.add_action(mecanum_controller_node)
    ld.add_action(laser_merger_node)
    ld.add_action(nav2_bringup_launch)
    ld.add_action(drl_inference_node)
    ld.add_action(rviz_node)
    return ld
