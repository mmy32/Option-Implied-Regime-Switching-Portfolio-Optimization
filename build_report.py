#!/usr/bin/env python3
"""
Pipeline script to compile the project report (report/main.tex) into a PDF.

Features:
- Automatically syncs generated figures (fig_01 through fig_06) from data/ or scripts/ into report/
- Compiles LaTeX using latexmk (or falls back to multi-pass pdflatex + bibtex)
- Resolves citations and cross-references (TOC, figures, tables)
- Supports optional cleanup of intermediate LaTeX build artifacts
"""

import os
import sys
import shutil
import subprocess
import argparse

def get_project_dirs():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(root_dir, "report")
    data_dir = os.path.join(root_dir, "data")
    return root_dir, report_dir, data_dir

def sync_figures(report_dir, data_dir):
    """Ensure all required figure assets are in report/."""
    figure_names = [
        "fig_01_market_regimes.png",
        "fig_02_drawdowns.png",
        "fig_03_rolling_volatility.png",
        "fig_04_weight_allocations.png",
        "fig_05_expanded_baselines.png",
        "fig_06_sensitivity.png",
    ]
    synced = []
    for fig in figure_names:
        src = os.path.join(data_dir, fig)
        dst = os.path.join(report_dir, fig)
        if os.path.exists(src):
            if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                shutil.copy2(src, dst)
                synced.append(fig)
        elif not os.path.exists(dst):
            print(f"[WARNING] Figure not found in data or report dir: {fig}")
    if synced:
        print(f"[INFO] Synced {len(synced)} figure(s) from {data_dir} to {report_dir}: {', '.join(synced)}")
    else:
        print(f"[INFO] All 6 figure assets are present and up-to-date in {report_dir}.")

def clean_intermediates(report_dir):
    """Remove LaTeX temporary build artifacts."""
    extensions = [
        ".aux", ".bbl", ".blg", ".log", ".out", ".toc",
        ".fls", ".fdb_latexmk", ".synctex.gz"
    ]
    cleaned = 0
    for fname in os.listdir(report_dir):
        _, ext = os.path.splitext(fname)
        if ext in extensions:
            path = os.path.join(report_dir, fname)
            try:
                os.remove(path)
                cleaned += 1
            except OSError as e:
                print(f"[WARN] Failed to remove {path}: {e}")
    print(f"[INFO] Cleaned {cleaned} intermediate LaTeX build file(s).")

def compile_pdf(report_dir, clean_after=False):
    """Compile main.tex into main.pdf."""
    tex_file = "main.tex"
    pdf_file = os.path.join(report_dir, "main.pdf")
    
    if not os.path.exists(os.path.join(report_dir, tex_file)):
        print(f"[ERROR] {tex_file} not found in {report_dir}")
        sys.exit(1)
        
    has_latexmk = shutil.which("latexmk") is not None
    has_pdflatex = shutil.which("pdflatex") is not None

    if not has_pdflatex and not has_latexmk:
        print("[ERROR] Neither 'latexmk' nor 'pdflatex' found on PATH. Please install TeX Live / MacTeX / TinyTeX.")
        sys.exit(1)

    print("[INFO] Starting LaTeX compilation pipeline...")
    
    if has_latexmk:
        print("[INFO] Using latexmk for automated multi-pass compilation...")
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", tex_file]
        res = subprocess.run(cmd, cwd=report_dir, capture_output=True, text=True)
        if res.returncode != 0:
            print("[ERROR] latexmk failed:")
            print(res.stderr or res.stdout)
            sys.exit(1)
    else:
        print("[INFO] Using multi-pass pdflatex + bibtex flow...")
        passes = [
            ["pdflatex", "-interaction=nonstopmode", tex_file],
            ["bibtex", "main"],
            ["pdflatex", "-interaction=nonstopmode", tex_file],
            ["pdflatex", "-interaction=nonstopmode", tex_file],
        ]
        for step, cmd in enumerate(passes, 1):
            print(f"  Step {step}/4: {' '.join(cmd)}")
            res = subprocess.run(cmd, cwd=report_dir, capture_output=True, text=True)
            if res.returncode != 0 and step == 1 and not os.path.exists(os.path.join(report_dir, "main.aux")):
                print(f"[ERROR] Step {step} failed:")
                print(res.stderr or res.stdout)
                sys.exit(1)

    if os.path.exists(pdf_file):
        size_kb = os.path.getsize(pdf_file) / 1024
        print(f"\n[SUCCESS] Successfully compiled: {pdf_file}")
        print(f"[SUCCESS] PDF file size: {size_kb:.1f} KB")
    else:
        print("[ERROR] PDF was not generated.")
        sys.exit(1)

    if clean_after:
        clean_intermediates(report_dir)

def main():
    parser = argparse.ArgumentParser(description="Compile report/main.tex into main.pdf with figures and bibliography.")
    parser.add_argument("--clean", action="store_true", help="Remove intermediate LaTeX build files after compilation.")
    parser.add_argument("--clean-only", action="store_true", help="Only remove intermediate build files and exit.")
    args = parser.parse_args()

    root_dir, report_dir, data_dir = get_project_dirs()

    if args.clean_only:
        clean_intermediates(report_dir)
        return

    sync_figures(report_dir, data_dir)
    compile_pdf(report_dir, clean_after=args.clean)

if __name__ == "__main__":
    main()
