#!/usr/bin/env python3
"""Zip the project with the files at the top level of the archive (no wrapper folder), ready to push to GitHub."""
import os, zipfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".venv", "__pycache__", ".git", ".zip_tmp"}; SKIP_FILES = {"bench"}
out = os.path.join(ROOT, "cuda_softmax_fusion.zip")
if os.path.exists(out): os.remove(out)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            p = os.path.join(d, f); rel = os.path.relpath(p, ROOT)
            if f.endswith(".zip") or (f in SKIP_FILES and d == ROOT): continue
            z.write(p, rel)
print("zip written:", out)
