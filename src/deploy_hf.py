"""Deploy the Q3B web app to a Hugging Face Docker Space.

Prereq (one-time, in a real terminal): authenticate with a WRITE token:
    python -c "from huggingface_hub import login; login()"
(paste the token from https://huggingface.co/settings/tokens)

Then run:  python src/deploy_hf.py
Uploads only what the app needs (app/, src/, Dockerfile, static assets incl. the
indexed DB) — the raw mp3s, notebooks, reports and figures are excluded.
"""
import os
import sys
import tempfile

from huggingface_hub import HfApi, create_repo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE_NAME = "ee200-audio-fingerprinting"
GITHUB = "https://github.com/Mayurdp2403/ee200-course-project"

README = f"""---
title: EE200 Audio Fingerprinting
emoji: "\U0001F3B5"
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# EE200: Audio Fingerprinting
Shazam-style audio identifier (Flask) for the EE200 course project, IIT Kanpur.
Library / Identify (upload, sample, or live mic) / Batch, plus a Robustness Lab
and a single-peak-vs-paired-hash comparison.

Source code: {GITHUB}
"""

IGNORE = [
    ".git*", ".git/**", "data/**", "notebooks/**", "report/**",
    "figures/**", "**/__pycache__/**", "*.pyc", "README.md", ".venv/**",
]


def main():
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        print("NOT LOGGED IN. First run, in a real terminal:\n"
              '  python -c "from huggingface_hub import login; login()"\n'
              "and paste a WRITE token from https://huggingface.co/settings/tokens")
        sys.exit(1)

    repo_id = f"{user}/{SPACE_NAME}"
    print(f"Authenticated as {user}. Creating/Updating Space: {repo_id}")
    create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    print("Uploading project files (excluding mp3s/notebooks/reports/figures)...")
    api.upload_folder(folder_path=ROOT, repo_id=repo_id, repo_type="space",
                      ignore_patterns=IGNORE, commit_message="Deploy Q3B web app")

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(README); readme_path = f.name
    api.upload_file(path_or_fileobj=readme_path, path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space",
                    commit_message="Space README + Docker config")
    os.unlink(readme_path)

    url = f"https://huggingface.co/spaces/{repo_id}"
    print("\nDONE. Space:", url)
    print("It will build the Docker image now (a few minutes). Live app URL:")
    print(f"  https://{user.lower()}-{SPACE_NAME}.hf.space")


if __name__ == "__main__":
    main()
