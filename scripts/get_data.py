"""Download the UCI Bank Marketing data and extract bank-full.csv.

Run from the repository root:
    python scripts/get_data.py

The official zip contains two inner zips: bank.zip (the 17-column data we use)
and bank-additional.zip (a different 21-column version we do NOT use).
"""
import io
import pathlib
import sys
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
OUT = pathlib.Path("data/raw")
TARGET = OUT / "bank-full.csv"
EXPECTED_ROWS = 45211  # data rows, header excluded


def main():
    if TARGET.exists():
        print("already present:", TARGET)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        print("downloading", URL)
        with urllib.request.urlopen(URL, timeout=60) as resp:
            outer = zipfile.ZipFile(io.BytesIO(resp.read()))
        inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
        inner.extract("bank-full.csv", OUT)
        print("saved", TARGET)

    # Sanity check: right file, right size (guards against bank-additional by mistake)
    with open(TARGET, encoding="utf-8") as f:
        header = f.readline()
        n_rows = sum(1 for _ in f)
    if n_rows != EXPECTED_ROWS or '"duration"' not in header:
        sys.exit(f"unexpected file: {n_rows} rows, header={header.strip()}")
    print(f"ok: {n_rows} rows, 17 columns")


if __name__ == "__main__":
    main()
