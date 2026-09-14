import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_amr_nav2_drl = get_package_share_directory('amr_nav2_drl')
    pkg_gazebo_sim = get_package_share_directory('gazebo_sim')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    use_sim_time_param = {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}

    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    use_laser_merger = LaunchConfiguration('use_laser_merger')

    default_map = os.path.join(pkg_gazebo_sim, 'maps', 'map.yaml')
    default_params = os.path.join(pkg_amr_nav2_drl, 'config', 'nav2_params_navfn_dwb.yaml')
    rviz_config = os.path.join(pkg_nav2_bringup, 'rviz', 'nav2_default_view.rviz')

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='false'))
    ld.add_action(DeclareLaunchArgument('map', default_value=default_map))
    ld.add_action(DeclareLaunchArgument('params_file', default_value=default_params))
    ld.add_action(DeclareLaunchArgument('use_laser_merger', default_value='true'))

    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_amr_nav2_drl, 'launch', 'bringup_launch.py')),
        launch_arguments={
            'map': map_yaml,
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': 'true',
            'use_composition': 'False',
        }.items(),
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

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[use_sim_time_param],
    )

    ld.add_action(nav2_bringup_launch)
    ld.add_action(laser_merger_node)
    ld.add_action(rviz_node)
    return ld
