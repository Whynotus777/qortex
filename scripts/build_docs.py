"""Build the docs site into ./site. Used locally, in Docker and as the Vercel build command."""
import os
import subprocess
import sys

root = os.path.join(os.path.dirname(__file__), "..")
subprocess.check_call([sys.executable, os.path.join(root, "scripts", "gen_api_docs.py")], cwd=root)
subprocess.check_call([sys.executable, "-m", "mkdocs", "build", "-q"], cwd=root)
print("docs built into site/")
