from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        # Output path format: <output_dir>/<controller_name>/<route_index>/<controller_name>-<trial_index>.csv
        # Example: DWB/1/DWB-1.csv. If the file already exists, trial_index auto-increments.
        DeclareLaunchArgument('controller_name', default_value='DWB'),
        DeclareLaunchArgument('run_id', default_value=''),
        DeclareLaunchArgument('route_index', default_value='1'),
        DeclareLaunchArgument('trial_index', default_value='1'),

        DeclareLaunchArgument('goal_x', default_value='2.150'),#2.150#1.0
        DeclareLaunchArgument('goal_y', default_value='9.604'),#9.604#-3.0
        DeclareLaunchArgument('goal_yaw', default_value='-0.606'),#-0.606#-1.5
        DeclareLaunchArgument('goal_frame', default_value='map'),

        DeclareLaunchArgument('navigate_action', default_value='navigate_to_pose'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('base_frame', default_value='base_link'),

        DeclareLaunchArgument('sample_rate_hz', default_value='20.0'),
        DeclareLaunchArgument('max_wait_action_server_sec', default_value='30.0'),
        DeclareLaunchArgument('movement_start_speed_threshold', default_value='0.003'),
        DeclareLaunchArgument('movement_start_wz_threshold', default_value='0.003'),
        DeclareLaunchArgument('stopped_speed_threshold', default_value='0.003'),
        DeclareLaunchArgument('stopped_wz_threshold', default_value='0.003'),
        DeclareLaunchArgument('stop_duration_sec', default_value='0.3'),
        DeclareLaunchArgument('failure_stop_duration_sec', default_value='5.0'),

        DeclareLaunchArgument('show_plots', default_value='true'),
        DeclareLaunchArgument('save_files', default_value='true'),
        DeclareLaunchArgument('output_dir', default_value='~/sac_ws/amr_nav2_experiments'),

        Node(
            package='amr_nav2_drl',
            executable='nav2_experiment_runner.py',
            name='nav2_experiment_runner',
            output='screen',
            parameters=[{
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),

                'controller_name': LaunchConfiguration('controller_name'),
                'run_id': LaunchConfiguration('run_id'),
                'route_index': ParameterValue(LaunchConfiguration('route_index'), value_type=int),
                'trial_index': ParameterValue(LaunchConfiguration('trial_index'), value_type=int),

                'goal_x': ParameterValue(LaunchConfiguration('goal_x'), value_type=float),
                'goal_y': ParameterValue(LaunchConfiguration('goal_y'), value_type=float),
                'goal_yaw': ParameterValue(LaunchConfiguration('goal_yaw'), value_type=float),
                'goal_frame': LaunchConfiguration('goal_frame'),

                'navigate_action': LaunchConfiguration('navigate_action'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'base_frame': LaunchConfiguration('base_frame'),

                'sample_rate_hz': ParameterValue(LaunchConfiguration('sample_rate_hz'), value_type=float),
                'max_wait_action_server_sec': ParameterValue(
                    LaunchConfiguration('max_wait_action_server_sec'),
                    value_type=float
                ),
                'movement_start_speed_threshold': ParameterValue(
                    LaunchConfiguration('movement_start_speed_threshold'),
                    value_type=float
                ),
                'movement_start_wz_threshold': ParameterValue(
                    LaunchConfiguration('movement_start_wz_threshold'),
                    value_type=float
                ),
                'stopped_speed_threshold': ParameterValue(
                    LaunchConfiguration('stopped_speed_threshold'),
                    value_type=float
                ),
                'stopped_wz_threshold': ParameterValue(
                    LaunchConfiguration('stopped_wz_threshold'),
                    value_type=float
                ),
                'stop_duration_sec': ParameterValue(LaunchConfiguration('stop_duration_sec'), value_type=float),
                'failure_stop_duration_sec': ParameterValue(
                    LaunchConfiguration('failure_stop_duration_sec'),
                    value_type=float
                ),

                'show_plots': ParameterValue(LaunchConfiguration('show_plots'), value_type=bool),
                'save_files': ParameterValue(LaunchConfiguration('save_files'), value_type=bool),
                'output_dir': LaunchConfiguration('output_dir'),
            }],
        ),
    ])