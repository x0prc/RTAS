from setuptools import setup, find_packages

setup(
    name="rtas",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchaudio",
        "PyQt5",
        "matplotlib",
        "sounddevice",
        "librosa",
        "scipy",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "stegano-cli=main:main",
        ],
        "gui_scripts": [
            "stegano-gui=gui.main:main",
        ]
    },
    package_data={},
    include_package_data=True,
)
