from setuptools import setup, find_packages

setup(
    name="tt_env",
    packages=find_packages(),
    package_data={'tt_env': ['robot/robot.urdf', 'table/robot.urdf', 'ball/robot.urdf']},
    version="0.0.1",
)