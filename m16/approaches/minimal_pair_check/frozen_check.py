"""Import T10's estimator from frozen/ after asserting the pre-registered sha256 of every file it uses."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHA = {"estimator.py": "fb7902e12240acd86dc6cd18786018abe73201a1996357efead379bd487fa867",
       "core.py": "48dbba49ab13367615e63a396d56f506bba9f8adda76ced4886a6023ecef8283",
       "walk.py": "8cdf611a37348f87b4c6947710ffd52f3f89286d52262884870d5f763066344b",
       "hom.py": "a91f019fca215b5b95996f77ff514811679ffda645112f010a0ec780a78b446d"}
for f, h in SHA.items():
    got = hashlib.sha256(open(os.path.join(HERE, "frozen", f), "rb").read()).hexdigest()
    assert got == h, f"frozen/{f} changed: {got}"
sys.path.insert(0, os.path.join(HERE, "frozen"))
import estimator as E  # noqa: E402,F401
