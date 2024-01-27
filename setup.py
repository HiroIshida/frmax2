from setuptools import setup

setup_requires = []

install_requires = [
    "numpy",
    "cmaes",
    "botorch",
    "dill",
]

setup(
    name="frmax2",
    version="0.0.0",
    description="feasible region maximization",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    package_data={"frmax": ["py.typed"]},
)
