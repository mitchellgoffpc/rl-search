#!/usr/bin/env python
import os
import glob
import shutil

if __name__ == "__main__":
    for f in glob.glob("*/episodes-*"):
        print(f"Removing {f}")
        shutil.rmtree(f)
    for f in glob.glob("*/*.ckpt"):
        print(f"Removing {f}")
        os.remove(f)
