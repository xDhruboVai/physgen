"""Markdown -> styled HTML -> PDF (Edge headless). Usage: python to_pdf.py <out_dir> <file.md> [...]"""
import os
import shutil
import subprocess
import sys

import markdown

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; margin: 0 0 10px; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 13.5pt; margin: 18px 0 6px; color: #222; }
table { border-collapse: collapse; margin: 8px 0 12px; font-size: 9pt; width: auto; }
th, td { border: 1px solid #bbb; padding: 3px 7px; text-align: left; vertical-align: top; }
th { background: #eef2f6; }
tr:nth-child(even) td { background: #fafafa; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f2f2f2; padding: 0 3px; }
img { max-width: 75%; display: block; margin: 8px auto; }
p em { color: #444; }
"""


def convert(md_path, out_dir):
    with open(md_path, encoding="utf-8") as f:
        body = markdown.markdown(f.read(), extensions=["tables"])
    stem = os.path.splitext(os.path.basename(md_path))[0]
    for img in ("demo_still.png",):  # images referenced by the documents
        src = os.path.join(os.path.dirname(md_path), img)
        if img in body and os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, img))
    html = os.path.join(out_dir, stem + ".html")
    with open(html, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
    pdf = os.path.join(out_dir, stem + ".pdf")
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    "file:///" + os.path.abspath(html).replace("\\", "/")], check=True, capture_output=True, timeout=120)
    os.remove(html)
    print("wrote", pdf)


if __name__ == "__main__":
    for md in sys.argv[2:]:
        convert(md, sys.argv[1])
