from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='tweet_911',
      version="0.0.1",
      description="911 Tweet Model (api_pred)",
      license="MIT",
      author="911 Team",
    #   author_email="contact@lewagon.org",
      url="https://github.com/Madeschaud/911-tweet.git",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
