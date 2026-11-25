"""Upload runs/ to the Hugging Face dataset (only new or changed files are sent).

Usage: upload_runs.py [-m MESSAGE]
Refuses to run unless the logged-in HF account is the expected personal one.
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from so101_zero_shot.config import RUNS_DIR

REPO_ID = "gilbertgonz/so101_zero_shot"
EXPECTED_USER = "gilbertgonz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--message", default="update runs")
    args = ap.parse_args()
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except (HfHubHTTPError, OSError) as e:
        raise SystemExit(f"not logged in to Hugging Face ({e}); run: hf auth login")
    if user != EXPECTED_USER:
        raise SystemExit(f"logged in to Hugging Face as {user!r}, expected {EXPECTED_USER!r}; refusing to upload")
    if not RUNS_DIR.is_dir():
        raise SystemExit(f"{RUNS_DIR} does not exist")
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    commit = api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=RUNS_DIR, commit_message=args.message)
    print(f"uploaded {RUNS_DIR} -> https://huggingface.co/datasets/{REPO_ID} ({commit.oid[:10]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
