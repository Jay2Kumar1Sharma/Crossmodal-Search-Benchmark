"""
download_data.py
================
Download Flickr8k from Kaggle. You only need this for LOCAL/Colab runs.

On Kaggle itself, DON'T run this — instead click  Add Input → search
"flickr8k" → adityajn105/flickr8k. It mounts read-only at /kaggle/input/flickr8k.

------------------------------------------------------------------
Kaggle API auth (one-time, for local/Colab)
------------------------------------------------------------------
1. kaggle.com → your avatar → Settings → API → "Create New Token".
   This downloads kaggle.json (contains your username + key).
2. Place it where the API looks for it:
       Linux/Mac : mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
       Windows   : move kaggle.json %USERPROFILE%\\.kaggle\\kaggle.json
   (Colab: upload kaggle.json, then the snippet below copies it into place.)
3. pip install kaggle
4. python data/download_data.py --out ./data/flickr8k
------------------------------------------------------------------
"""
import argparse
import os
import subprocess
import sys
import zipfile


DATASET = "adityajn105/flickr8k"


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return os.path.isfile(os.path.expanduser("~/.kaggle/kaggle.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/flickr8k", help="output directory")
    args = ap.parse_args()

    if not have_credentials():
        sys.exit(
            "No Kaggle credentials found.\n"
            "Set up ~/.kaggle/kaggle.json (see the instructions at the top of this file),\n"
            "or export KAGGLE_USERNAME and KAGGLE_KEY, then re-run."
        )

    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading {DATASET} -> {args.out} ...")
    # Uses the official Kaggle CLI; --unzip extracts in place.
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", args.out, "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        sys.exit("`kaggle` CLI not found. Run: pip install kaggle")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Kaggle download failed (auth or network?). Details: {e}")

    # Sanity check expected layout.
    images = os.path.join(args.out, "Images")
    caps = os.path.join(args.out, "captions.txt")
    ok = os.path.isdir(images) and os.path.isfile(caps)
    print("OK" if ok else "WARNING: expected Images/ and captions.txt not found",
          f"\n  Images/ exists : {os.path.isdir(images)}",
          f"\n  captions.txt   : {os.path.isfile(caps)}")
    if ok:
        n = len([f for f in os.listdir(images) if f.lower().endswith('.jpg')])
        print(f"  {n} images downloaded.")


if __name__ == "__main__":
    main()
