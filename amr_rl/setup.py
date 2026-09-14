from setuptools import find_packages, setup

package_name = 'amr_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robert',
    maintainer_email='robert@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'train_stage0 = amr_rl.train_stage0:main',
        'train_stage1 = amr_rl.train_stage1:main',
        'train_stage2 = amr_rl.train_stage2:main',
        'train_stage3 = amr_rl.train_stage3:main',
        'train_stage3_1 = amr_rl.train_stage3_1:main',
        'eval_stage0 = amr_rl.eval_stage0:main',
        'eval_stage1 = amr_rl.eval_stage1:main',
        ],
    },
)
