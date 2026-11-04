"""Build report/report.pdf from report/report.md.

    python scripts/build_report.py

Converts the Markdown to HTML (Python-Markdown) and prints it to PDF with a headless
Chrome or Edge. Figures are read from results/figures/, so run src.evaluate first.
Not part of the experiment pipeline: it only typesets the write-up.
"""
import os
import pathlib
import shutil
import subprocess
import sys

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
SRC, PDF = REPORT / "report.md", REPORT / "report.pdf"
TMP_HTML = REPORT / "_report_build.html"   # must sit next to report.md so ../results/ paths resolve

CSS = """
@page { size: A4; margin: 17mm 17mm 16mm 17mm;
        @bottom-center { content: counter(page); font: 9pt "Segoe UI", Arial, sans-serif; color: #666; } }
body { font: 10.3pt/1.42 Georgia, "Times New Roman", serif; color: #1a1a1a; }
h1 { font: 600 17pt/1.2 "Segoe UI", Arial, sans-serif; margin: 0 0 6px; }
h2 { font: 600 12.5pt "Segoe UI", Arial, sans-serif; margin: 14px 0 4px; border-bottom: 1px solid #ccc;
     padding-bottom: 2px; break-after: avoid; }
p { margin: 5px 0; text-align: justify; hyphens: auto; }
ul, ol { margin: 4px 0; padding-left: 20px; } li { margin: 2px 0; text-align: justify; }
code { font: 9pt Consolas, monospace; background: #f2f2f2; padding: 0 2px; border-radius: 2px; }
.titleblock { text-align: center; margin-bottom: 8px; }
.titleblock p { text-align: center; font: 9.5pt/1.5 "Segoe UI", Arial, sans-serif; color: #333; }
.abstract { background: #f5f7fa; border-left: 3px solid #2a78d6; padding: 6px 12px; margin: 8px 0 10px; }
table { border-collapse: collapse; width: 100%; margin: 4px 0 8px; break-inside: avoid;
        font: 8.6pt/1.3 "Segoe UI", Arial, sans-serif; }
th { background: #eef1f5; text-align: left; font-weight: 600; }
th, td { border-bottom: 1px solid #d5d9df; padding: 3px 5px; }
.table-caption { font: 8.8pt/1.35 "Segoe UI", Arial, sans-serif; color: #333; margin-top: 8px; break-after: avoid; }
figure { margin: 6px 0; break-inside: avoid; }
figure img { width: 100%; display: block; }
figcaption { font: 8.6pt/1.35 "Segoe UI", Arial, sans-serif; color: #444; margin-top: 3px; }
/* Each image's flex = its width/height, so both images in a row get the same height. */
.figrow { display: flex; gap: 10px; align-items: flex-start; }
.figrow img { flex-basis: 0; min-width: 0; width: 0; }
figure.wide img { width: 88%; margin: 0 auto; }
.refs ol { font-size: 9pt; }
"""


def browser():
    for name in ("chrome", "msedge", "google-chrome", "chromium", "chromium-browser"):
        if shutil.which(name):
            return shutil.which(name)
    for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
        if os.path.exists(p):
            return p
    sys.exit("Chrome or Edge not found; install one or open report/report.md and print it to PDF.")


def main():
    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables", "md_in_html"])
    TMP_HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>CallWise report</title>"
                        f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    try:
        subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={PDF}", TMP_HTML.as_uri()],
                       check=True, capture_output=True, timeout=120)
    finally:
        TMP_HTML.unlink(missing_ok=True)
    print("saved", PDF.relative_to(ROOT))


if __name__ == "__main__":
    main()
