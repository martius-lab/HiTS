from setuptools import setup, find_packages

setup(name="hac_envs",
      packages=find_packages(),
      version="1.0",
      install_requires=["numpy", "gym", "mujoco-py==2.0.2.13"],
      include_package_data=True
)
