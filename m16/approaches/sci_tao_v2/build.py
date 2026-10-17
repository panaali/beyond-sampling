"""Build the optional backend with a C++ compiler; never downloads dependencies."""
from pathlib import Path
import subprocess

if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    subprocess.run(['g++', '-std=c++17', '-O3', '-ffp-contract=off', '-fPIC', '-shared',
                    str(folder / 'walker.cpp'), '-o', str(folder / '_walker.so')], check=True)
    print(folder / '_walker.so')
