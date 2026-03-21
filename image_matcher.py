#!/usr/bin/env python3
"""
Image Matcher GUI  —  Tiered 3-Phase Matching
----------------------------------------------
Designed for the real-world case where many similar originals exist and
one was selected/cropped/resized for publishing.

THREE PHASES (run in sequence or all at once):
  Phase 1 — EXACT  : pHash distance 0  (truly identical content, 100%)
  Phase 2 — CLOSE  : pHash distance 1-6  (same shot, minor re-save/crop)
  Phase 3 — FUZZY  : pHash distance 7-N  (similar composition, slider controls N)

Unmatched images from Phase 1 are passed to Phase 2, then to Phase 3.
Already-matched images are never re-processed in later phases.

Card colours:
  Gold  border  = Phase 1 exact match  (100%)
  Green border  = Phase 2 close match  (>=90%)
  Teal  border  = Phase 3 fuzzy match  (slider range)
  Red   border  = not found

Requirements:
    pip install Pillow imagehash

Usage:
    python image_matcher.py
"""

import sys
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from PIL import Image, ImageTk
    import imagehash
except ImportError:
    print("Missing dependencies. Run:  pip install Pillow imagehash")
    sys.exit(1)

# OpenCV is optional — used for ORB feature matching in Phase 3
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


# ── cross-platform font resolver ──────────────────────────────────────────
def _resolve_fonts():
    """
    Pick the best available monospace font on this platform.
    Prefers Consolas (Windows) → DejaVu Sans Mono (Linux) → Courier New → Courier.
    Returns a dict of (family, size[, style]) tuples ready for tkinter font= args.
    """
    import tkinter.font as tkfont
    # tkfont.families() only works after a Tk root exists, so we use a temp root
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        available = set(tkfont.families(_root))
        _root.destroy()
    except Exception:
        available = set()

    # Ordered preference list
    mono_candidates = [
        "Consolas",           # Windows (Anaconda/Miniconda)
        "DejaVu Sans Mono",   # Linux — confirmed working
        "Liberation Mono",    # Linux — confirmed working
        "Ubuntu Mono",        # Linux — confirmed working
        "Noto Mono",          # Linux — confirmed working
        "Noto Sans Mono",     # Linux variant
        "FreeMono",           # Linux fallback
        "Andale Mono",        # Linux (found via sudo)
        "Nimbus Mono PS",     # Linux fallback
        "Courier New",        # Windows/Mac
        "Courier",            # universal last resort
    ]
    mono = next((f for f in mono_candidates if f in available), "Courier")

    return {
        "title":   (mono, 14, "bold"),
        "label":   (mono,  8),
        "label_b": (mono,  8, "bold"),
        "entry":   (mono,  9),
        "section": (mono,  9, "bold"),
        "badge":   (mono,  8, "bold"),
        "name":    (mono,  7),
        "info":    (mono,  8),
        "status":  (mono,  9),
        "stats":   (mono,  8),
        "btn":     (mono,  9),
        "btn_run": (mono, 11, "bold"),
        "preview": (mono,  9),
        "dim":     (mono,  9),
    }

F = _resolve_fonts()


# ── colours & constants ────────────────────────────────────────────────────
THUMBNAIL_SIZE  = (148, 148)
GRID_COLS       = 6

BG_DARK   = "#0d0d1a"
BG_MID    = "#131325"
BG_CARD   = "#1a1a30"
BG_CARD_1 = "#2a220a"
BG_CARD_2 = "#0a2214"
BG_CARD_3 = "#072020"

C_GOLD    = "#ffd700"
C_GREEN   = "#00e676"
C_TEAL    = "#26c6da"
C_RED     = "#ff1744"
C_BLUE    = "#448aff"
C_TEXT    = "#dde1f0"
C_DIM     = "#7986a8"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

PHASE1_MAX = 0       # pHash+dHash distance: exact
PHASE2_MAX = 6       # pHash+dHash distance: close
ORB_MIN_MATCHES = 8  # minimum good ORB keypoint matches to count as fuzzy match
ORB_RESIZE = 800     # resize longest edge to this before ORB (speed vs accuracy)


# ── data model ─────────────────────────────────────────────────────────────
@dataclass
class ImageEntry:
    small_path:  Path
    phash:       object       = field(default=None, repr=False)
    thumbnail:   object       = field(default=None, repr=False)
    match_path:  Optional[Path] = None
    match_dist:  int          = 999
    match_score: float        = 0.0
    phase:       int          = 0

    @property
    def found(self):
        return self.phase > 0

    @property
    def border_color(self):
        return {1: C_GOLD, 2: C_GREEN, 3: C_TEAL}.get(self.phase, C_RED)

    @property
    def bg_color(self):
        return {1: BG_CARD_1, 2: BG_CARD_2, 3: BG_CARD_3}.get(self.phase, BG_CARD)

    @property
    def phase_label(self):
        return {1: "EXACT", 2: "CLOSE", 3: "FUZZY"}.get(self.phase, "NONE")


# ── matching engine ─────────────────────────────────────────────────────────
def collect_images(folder: Path):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXT
    )


def dual_hash(path: Path):
    """Return (phash, dhash) tuple, either may be None on failure."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            ph = imagehash.phash(img)
            dh = imagehash.dhash(img)
            return ph, dh
    except Exception:
        return None, None


def best_hash_dist(ph1, dh1, ph2, dh2):
    """Return the minimum of pHash and dHash distances (lower = more similar)."""
    dists = []
    if ph1 is not None and ph2 is not None:
        dists.append(ph1 - ph2)
    if dh1 is not None and dh2 is not None:
        dists.append(dh1 - dh2)
    return min(dists) if dists else 999


def dist_to_score(dist: int, max_dist: int = 64):
    return round(max(0.0, (1.0 - dist / max_dist)) * 100, 1)


def orb_load(path: Path):
    """Load image as grayscale numpy array for ORB, resized for speed."""
    if not OPENCV_AVAILABLE:
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("L")   # grayscale
            w, h = img.size
            scale = ORB_RESIZE / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            return np.array(img)
    except Exception:
        return None


def orb_match_score(small_arr, large_arr, min_matches: int = ORB_MIN_MATCHES):
    """
    Match keypoints between small and large image using ORB + BFMatcher.
    Returns a 0-100 score based on number of good matches, or 0 if insufficient.
    ORB is rotation/scale invariant — ideal for cropped/resized originals.
    """
    if small_arr is None or large_arr is None:
        return 0.0
    try:
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(small_arr, None)
        kp2, des2 = orb.detectAndCompute(large_arr, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        # Lowe's ratio test — keep only clearly better matches
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < min_matches:
            return 0.0
        # Score: ratio of good matches to total keypoints in smaller image
        score = min(100.0, (len(good) / len(kp1)) * 100 * 2)
        return round(score, 1)
    except Exception:
        return 0.0


def run_tiered_match(small_list, large_list, orb_min_matches, active_phases=None, progress_cb=None):
    """
    Phase 1 — Exact : pHash+dHash distance == 0
    Phase 2 — Close : pHash+dHash distance 1-6
    Phase 3 — Fuzzy : ORB feature matching (OpenCV) — handles crops/rescales
                      Falls back to loose pHash if OpenCV not available.

    Each original can only be claimed ONCE (1:1 matching).
    Phase 3 requires a meaningful ORB score (>= 15%) to avoid false positives.
    """
    if active_phases is None:
        active_phases = [1, 2, 3]

    # Track which originals have already been claimed (1 original → 1 thumbnail)
    claimed: set = set()

    # ── index originals ───────────────────────────────────────────
    large_hashes = []   # (path, phash, dhash)
    n = len(large_list)
    for i, lp in enumerate(large_list):
        ph, dh = dual_hash(lp)
        if ph is not None or dh is not None:
            large_hashes.append((lp, ph, dh))
        if progress_cb and i % 25 == 0:
            progress_cb(i, n, "Indexing originals")
    if progress_cb:
        progress_cb(n, n, "Indexing originals")

    # ── hash thumbnails ───────────────────────────────────────────
    entries = []
    n = len(small_list)
    for i, sp in enumerate(small_list):
        ph, dh = dual_hash(sp)
        entries.append(ImageEntry(small_path=sp, phash=(ph, dh)))
        if progress_cb and i % 10 == 0:
            progress_cb(i, n, "Hashing thumbnails")
    if progress_cb:
        progress_cb(n, n, "Hashing thumbnails")

    # ── phase 1 & 2: dual hash ────────────────────────────────────
    for phase, (lo, hi) in [(1, (0, PHASE1_MAX)), (2, (PHASE1_MAX+1, PHASE2_MAX))]:
        if phase not in active_phases:
            continue
        unmatched = [e for e in entries if not e.found]
        if not unmatched:
            break
        label = "Phase 1 — Exact" if phase == 1 else "Phase 2 — Close (pHash+dHash)"
        n = len(unmatched)
        for i, entry in enumerate(unmatched):
            eph, edh = entry.phash
            best_dist = 999
            best_path = None
            for lp, lph, ldh in large_hashes:
                if lp in claimed:
                    continue  # skip already claimed originals
                d = best_hash_dist(eph, edh, lph, ldh)
                if d < best_dist:
                    best_dist = d
                    best_path = lp
            if best_path is not None and lo <= best_dist <= hi:
                entry.match_path  = best_path
                entry.match_dist  = best_dist
                entry.match_score = dist_to_score(best_dist)
                entry.phase       = phase
                claimed.add(best_path)
            if progress_cb and i % 5 == 0:
                progress_cb(i, n, label)
        if progress_cb:
            progress_cb(n, n, label)

    # ── phase 3: ORB feature matching (or pHash fallback) ─────────
    unmatched = [e for e in entries if not e.found]
    if unmatched and 3 in active_phases:
        if OPENCV_AVAILABLE:
            label = "Phase 3 — Fuzzy (ORB features)"
            # Pre-load large images for ORB (skip already claimed)
            large_orb = [(lp, lph, ldh, orb_load(lp))
                         for lp, lph, ldh in large_hashes
                         if lp not in claimed]
            n = len(unmatched)
            for i, entry in enumerate(unmatched):
                small_arr = orb_load(entry.small_path)
                best_score = 0.0
                best_path  = None
                best_dist  = 999
                eph, edh = entry.phash
                for lp, lph, ldh, larr in large_orb:
                    if lp in claimed:
                        continue
                    score = orb_match_score(small_arr, larr, orb_min_matches)
                    if score > best_score:
                        best_score = score
                        best_path  = lp
                        best_dist  = best_hash_dist(eph, edh, lph, ldh)
                # Require at least 15% ORB score to avoid false positives
                if best_score >= 15.0:
                    entry.match_path  = best_path
                    entry.match_dist  = best_dist
                    entry.match_score = best_score
                    entry.phase       = 3
                    claimed.add(best_path)
                if progress_cb and i % 3 == 0:
                    progress_cb(i, n, label)
            if progress_cb:
                progress_cb(n, n, label)
        else:
            # Fallback: loose pHash pass with higher threshold
            label = "Phase 3 — Fuzzy (pHash fallback, no OpenCV)"
            n = len(unmatched)
            for i, entry in enumerate(unmatched):
                eph, edh = entry.phash
                best_dist = 999
                best_path = None
                for lp, lph, ldh in large_hashes:
                    if lp in claimed:
                        continue
                    d = best_hash_dist(eph, edh, lph, ldh)
                    if d < best_dist:
                        best_dist = d
                        best_path = lp
                if best_path is not None and PHASE2_MAX < best_dist <= 20:
                    entry.match_path  = best_path
                    entry.match_dist  = best_dist
                    entry.match_score = dist_to_score(best_dist)
                    entry.phase       = 3
                    claimed.add(best_path)
                if progress_cb and i % 5 == 0:
                    progress_cb(i, n, label)
            if progress_cb:
                progress_cb(n, n, label)

    return entries


# ── GUI ─────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Matcher  —  Tiered 3-Phase  (ORB+pHash)")
        self.geometry("1280x860")
        self.configure(bg=BG_DARK)
        self.minsize(960, 640)

        self.entries    = []
        self._photos    = []
        self._cards     = []
        self._filter    = tk.StringVar(value="all")
        self._p3_min_matches = tk.IntVar(value=8)
        self._run_phases = [tk.BooleanVar(value=True) for _ in range(3)]
        self._selected  = None

        self._status_var = tk.StringVar(value="Select folders, choose phases and click Run.")
        self._prog_var  = tk.DoubleVar()
        self._prog_lbl  = tk.StringVar(value="")

        self._small_dir = tk.StringVar()
        self._large_dir = tk.StringVar()
        self._dest_dir  = tk.StringVar()
        # Trace: when _small_dir is set (via entry or browse), load previews
        self._small_dir_trace_after = None
        self._small_dir.trace_add("write", self._on_small_dir_changed)

        self._build_ui()

    # ── build UI ────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_toolbar()

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg=BG_MID, width=280)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        self._build_sidebar(left)

        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_grid_area(right)

        tk.Label(self, textvariable=self._status_var,
                 bg=BG_MID, fg=C_DIM, anchor=tk.W,
                 padx=12, pady=3, font=F["entry"]
                 ).pack(fill=tk.X, side=tk.BOTTOM)

    # ── toolbar ─────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        bar = tk.Frame(self, bg=BG_MID, pady=10)
        bar.pack(fill=tk.X)

        tk.Label(bar, text="◈  IMAGE MATCHER", bg=BG_MID,
                 fg=C_BLUE, font=F["title"],
                 padx=14).pack(side=tk.LEFT)

        for icon, label, var in [
            ("🔍", "Thumbnails",  self._small_dir),
            ("🗂",  "Originals",   self._large_dir),
            ("📁", "Destination", self._dest_dir),
        ]:
            f = tk.Frame(bar, bg=BG_MID)
            f.pack(side=tk.LEFT, padx=8)
            tk.Label(f, text=f"{icon} {label}", bg=BG_MID,
                     fg=C_DIM, font=F["label"]).pack(anchor=tk.W)
            row = tk.Frame(f, bg=BG_MID)
            row.pack()
            tk.Entry(row, textvariable=var, width=24,
                     bg="#1c1c38", fg=C_TEXT, relief=tk.FLAT,
                     insertbackground=C_TEXT,
                     font=F["entry"]).pack(side=tk.LEFT)
            tk.Button(row, text="…", command=lambda v=var: self._browse(v),
                      bg="#28284a", fg=C_TEXT, relief=tk.FLAT,
                      padx=5, cursor="hand2").pack(side=tk.LEFT, padx=1)

        # Phase checkboxes
        pf = tk.Frame(bar, bg=BG_MID)
        pf.pack(side=tk.LEFT, padx=10)
        tk.Label(pf, text="Run phases:", bg=BG_MID,
                 fg=C_DIM, font=F["label"]).pack(anchor=tk.W)
        pb = tk.Frame(pf, bg=BG_MID)
        pb.pack()
        for i, (lbl, col) in enumerate([
                ("1 Exact", C_GOLD), ("2 Close", C_GREEN), ("3 Fuzzy", C_TEAL)]):
            tk.Checkbutton(pb, text=lbl, variable=self._run_phases[i],
                           bg=BG_MID, fg=col, selectcolor=BG_DARK,
                           activebackground=BG_MID,
                           font=F["label_b"]
                           ).pack(side=tk.LEFT, padx=3)

        # Phase-3 ORB sensitivity
        sf = tk.Frame(bar, bg=BG_MID)
        sf.pack(side=tk.LEFT, padx=4)
        tk.Label(sf, text="Phase 3 min matches:", bg=BG_MID,
                 fg=C_DIM, font=F["label"]).pack(anchor=tk.W)
        row2 = tk.Frame(sf, bg=BG_MID)
        row2.pack()
        tk.Label(row2, textvariable=self._p3_min_matches,
                 bg=BG_MID, fg=C_TEAL,
                 font=F["section"], width=3).pack(side=tk.LEFT)
        tk.Scale(row2, from_=4, to=30, orient=tk.HORIZONTAL,
                 variable=self._p3_min_matches,
                 bg=BG_MID, fg=C_TEXT, troughcolor="#222244",
                 highlightthickness=0, length=110, showvalue=False
                 ).pack(side=tk.LEFT)

        tk.Button(bar, text="▶  RUN",
                  command=self._start,
                  bg=C_BLUE, fg=BG_DARK,
                  font=F["btn_run"],
                  relief=tk.FLAT, padx=18, pady=7,
                  cursor="hand2").pack(side=tk.RIGHT, padx=16)
        # OpenCV availability indicator
        cv_text = "OpenCV ✓" if OPENCV_AVAILABLE else "OpenCV ✗ (pip install opencv-python)"
        cv_color = C_GREEN if OPENCV_AVAILABLE else C_RED
        tk.Label(bar, text=cv_text, bg=BG_MID, fg=cv_color,
                 font=F["label"]).pack(side=tk.RIGHT, padx=6)

    # ── sidebar ─────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        def section(text):
            tk.Label(parent, text=text, bg=BG_MID,
                     fg=C_BLUE, font=F["section"],
                     pady=6, padx=12, anchor=tk.W).pack(fill=tk.X)

        section("PREVIEW")
        self._preview = tk.Label(parent, bg=BG_MID,
                                  text="click a card",
                                  fg=C_DIM, font=F["btn"])
        self._preview.pack(pady=(0, 6))

        section("DETAILS")
        self._info = tk.Text(parent, bg="#0d0d1e", fg=C_TEXT,
                              font=F["info"], relief=tk.FLAT,
                              wrap=tk.WORD, height=13,
                              padx=8, pady=8, state=tk.DISABLED)
        self._info.pack(fill=tk.X, padx=8)

        section("LEGEND")
        for col, lbl in [
            (C_GOLD,  "Phase 1 — Exact  (100%)"),
            (C_GREEN, "Phase 2 — Close  (>=90%)"),
            (C_TEAL,  "Phase 3 — Fuzzy  (ORB features)"),
            (C_RED,   "Not matched"),
        ]:
            row = tk.Frame(parent, bg=BG_MID)
            row.pack(anchor=tk.W, padx=12, pady=1)
            tk.Label(row, text="█", fg=col, bg=BG_MID,
                     font=F["btn"]).pack(side=tk.LEFT)
            tk.Label(row, text=lbl, fg=C_DIM, bg=BG_MID,
                     font=F["label"]).pack(side=tk.LEFT, padx=4)

        section("FILTER")
        for val, lbl in [
            ("all",      "All"),
            ("p1",       "Exact (gold)"),
            ("p2",       "Close (green)"),
            ("p3",       "Fuzzy (teal)"),
            ("notfound", "Not found (red)"),
        ]:
            tk.Radiobutton(parent, text=lbl,
                           variable=self._filter, value=val,
                           command=self._apply_filter,
                           bg=BG_MID, fg=C_TEXT, selectcolor=BG_DARK,
                           activebackground=BG_MID,
                           font=F["btn"]).pack(anchor=tk.W, padx=16)

        # Remove matched thumbnails button — inline with filter
        tk.Button(parent, text="✕  Remove matched from list",
                  command=self._remove_matched,
                  bg="#2a1a1a", fg=C_RED, relief=tk.FLAT,
                  font=F["btn"], pady=5, cursor="hand2"
                  ).pack(fill=tk.X, padx=8, pady=(6, 2))

        section("STATS")
        self._stats = tk.Label(parent, text="—", bg=BG_MID,
                                fg=C_DIM, font=F["info"],
                                justify=tk.LEFT, padx=14)
        self._stats.pack(anchor=tk.W)

        section("ACTIONS")
        for icon, lbl, cmd in [
            ("📋", "Export list (.txt)",     self._export_txt),
            ("📂", "Copy originals to dest", self._copy_originals),
        ]:
            tk.Button(parent, text=f"{icon}  {lbl}", command=cmd,
                      bg="#1c1c38", fg=C_TEXT, relief=tk.FLAT,
                      font=F["btn"], pady=6,
                      cursor="hand2").pack(fill=tk.X, padx=8, pady=2)

    # ── grid area ────────────────────────────────────────────────────────────
    def _build_grid_area(self, parent):
        ph = tk.Frame(parent, bg=BG_DARK)
        ph.pack(fill=tk.X, padx=4, pady=(4, 0))
        tk.Label(ph, textvariable=self._prog_lbl,
                 bg=BG_DARK, fg=C_DIM,
                 font=F["info"]).pack(side=tk.LEFT, padx=4)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Horizontal.TProgressbar",
                         background=C_BLUE, troughcolor="#1c1c38", borderwidth=0)
        ttk.Progressbar(ph, variable=self._prog_var,
                         maximum=100, length=300).pack(side=tk.LEFT)

        container = tk.Frame(parent, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(container, bg=BG_DARK, highlightthickness=0)
        vsb = tk.Scrollbar(container, orient=tk.VERTICAL,
                           command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._gframe = tk.Frame(self._canvas, bg=BG_DARK)
        self._cwin = self._canvas.create_window(
            (0, 0), window=self._gframe, anchor=tk.NW)

        self._gframe.bind("<Configure>",
            lambda _e: self._canvas.configure(
                scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",
            lambda e: self._canvas.itemconfig(self._cwin, width=e.width))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._canvas.bind_all(seq, self._scroll)

    def _scroll(self, event):
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    # ── helpers ──────────────────────────────────────────────────────────────
    def _browse(self, var):
        p = filedialog.askdirectory()
        if p:
            var.set(p)
            # If thumbnails folder: preview loads via _small_dir trace automatically

    def _load_previews(self, folder: Path):
        """Show all thumbnail images immediately as unmatched cards (grey),
        before any matching has been run."""
        self._clear_grid()
        self.entries = []
        self._status_var.set(f"Loading thumbnails from {folder.name} …")

        def worker():
            paths = collect_images(folder)
            if not paths:
                self.after(0, lambda: self._status_var.set("No images found in that folder."))
                return
            entries = [ImageEntry(small_path=p) for p in paths]
            self.after(0, lambda: self._status_var.set(
                f"{len(entries)} thumbnails loaded — set Originals folder and click Run."))
            self.after(0, lambda: self._show_preview_entries(entries))

        threading.Thread(target=worker, daemon=True).start()

    def _show_preview_entries(self, entries):
        self.entries = entries
        self._render_grid(entries)
        self._update_stats()

    def _on_small_dir_changed(self, *_):
        """Debounced: fires when _small_dir changes (browse or typed)."""
        # Cancel any pending call so rapid typing doesn't thrash
        if self._small_dir_trace_after:
            self.after_cancel(self._small_dir_trace_after)
        def check():
            self._small_dir_trace_after = None
            p = Path(self._small_dir.get())
            if p.is_dir():
                self._load_previews(p)
        self._small_dir_trace_after = self.after(400, check)

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    # ── run ──────────────────────────────────────────────────────────────────
    def _start(self):
        sd = Path(self._small_dir.get())
        ld = Path(self._large_dir.get())
        if not sd.is_dir():
            messagebox.showerror("Error", "Thumbnails folder not found.")
            return
        if not ld.is_dir():
            messagebox.showerror("Error", "Originals folder not found.")
            return

        active = [i+1 for i in range(3) if self._run_phases[i].get()]
        if not active:
            messagebox.showwarning("No phases", "Enable at least one phase.")
            return

        self._clear_grid()
        self._prog_var.set(0)
        orb_min = self._p3_min_matches.get()

        def worker():
            self.after(0, lambda: self._set_status("Scanning…"))
            small = collect_images(sd)
            large = collect_images(ld)
            if not small:
                self.after(0, lambda: messagebox.showinfo("Info", "No thumbnails found."))
                return
            self.after(0, lambda: self._set_status(
                f"{len(small)} thumbnails  ·  {len(large)} originals  ·  matching…"))

            def prog(cur, total, label):
                pct = (cur / max(total, 1)) * 100
                self.after(0, lambda: self._prog_var.set(pct))
                self.after(0, lambda: self._prog_lbl.set(f"{label}  {cur}/{total}"))

            entries = run_tiered_match(small, large, orb_min, active, prog)

            self.after(0, lambda: self._done(entries))

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, entries):
        self.entries = entries
        self._prog_var.set(100)
        p1 = sum(1 for e in entries if e.phase == 1)
        p2 = sum(1 for e in entries if e.phase == 2)
        p3 = sum(1 for e in entries if e.phase == 3)
        nm = len(entries) - p1 - p2 - p3
        self._set_status(
            f"Done  ·  {len(entries)} total  |  "
            f"Exact={p1}  Close={p2}  Fuzzy={p3}  Unmatched={nm}")
        self._update_stats()
        self._render_grid(entries)

    # ── grid ─────────────────────────────────────────────────────────────────
    def _clear_grid(self):
        for w in self._gframe.winfo_children():
            w.destroy()
        self._photos.clear()
        self._cards.clear()

    def _render_grid(self, entries):
        self._clear_grid()
        for idx, entry in enumerate(entries):
            r, c = divmod(idx, GRID_COLS)
            card = self._make_card(entry)
            card.grid(row=r, column=c, padx=5, pady=5, sticky=tk.NSEW)
            self._cards.append(card)
        self._canvas.yview_moveto(0)

    def _make_card(self, entry: ImageEntry):
        bg = entry.bg_color
        card = tk.Frame(self._gframe, bg=bg,
                        highlightbackground=entry.border_color,
                        highlightthickness=3, cursor="hand2")

        try:
            with Image.open(entry.small_path) as raw:
                img = raw.convert("RGB")
            img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
            # Parse bg hex to RGB tuple
            bghex = bg.lstrip("#")
            bgrgb = tuple(int(bghex[i:i+2], 16) for i in (0, 2, 4))
            canvas_img = Image.new("RGB", THUMBNAIL_SIZE, bgrgb)
            off = ((THUMBNAIL_SIZE[0]-img.width)//2,
                   (THUMBNAIL_SIZE[1]-img.height)//2)
            canvas_img.paste(img, off)
            photo = ImageTk.PhotoImage(canvas_img)
        except Exception:
            photo = ImageTk.PhotoImage(Image.new("RGB", THUMBNAIL_SIZE, (30, 30, 50)))

        self._photos.append(photo)
        entry.thumbnail = photo

        lbl = tk.Label(card, image=photo, bg=bg)
        lbl.pack()

        if entry.found:
            badge_text  = f"[{entry.phase_label}]  {entry.match_score:.0f}%"
            badge_color = entry.border_color
        else:
            badge_text  = "x  not matched"
            badge_color = C_RED

        tk.Label(card, text=badge_text, bg=bg, fg=badge_color,
                 font=F["label_b"]).pack(pady=(2, 0))

        name = entry.small_path.name
        if len(name) > 19:
            name = name[:16] + "…"
        tk.Label(card, text=name, bg=bg, fg=C_DIM,
                 font=F["name"]).pack(pady=(0, 4))

        for w in (card, lbl):
            w.bind("<Button-1>", lambda _e, en=entry: self._click(en))
        return card

    # ── card click ───────────────────────────────────────────────────────────
    def _click(self, entry: ImageEntry):
        self._selected = entry
        try:
            with Image.open(entry.small_path) as raw:
                img = raw.convert("RGB")
            img.thumbnail((256, 192), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._preview.config(image=photo, text="")
            self._preview._ph = photo
        except Exception:
            self._preview.config(text="(error)")

        lines = [
            "THUMBNAIL",
            f"  {entry.small_path.name}",
            f"  {entry.small_path.parent}",
            "",
        ]
        if entry.found:
            lines += [
                f"STATUS  : {entry.phase_label}  (phase {entry.phase})",
                f"SCORE   : {entry.match_score:.1f}%",
                f"DISTANCE: {entry.match_dist}",
                "",
                "ORIGINAL",
                f"  {entry.match_path.name}",
                f"  {entry.match_path.parent}",
            ]
        else:
            lines += ["STATUS  : NOT MATCHED"]

        self._info.config(state=tk.NORMAL)
        self._info.delete("1.0", tk.END)
        self._info.insert(tk.END, "\n".join(lines))
        self._info.config(state=tk.DISABLED)

    # ── remove matched ───────────────────────────────────────────────────────
    def _remove_matched(self):
        """Move all matched thumbnails to a 'founded' folder next to the
        Thumbnails folder, then remove them from the list.
        Keeps only unmatched ones for the next Originals folder run."""
        matched = [e for e in self.entries if e.found]
        kept    = [e for e in self.entries if not e.found]
        if not matched:
            messagebox.showinfo("Nothing to remove", "No matched images in the list.")
            return

        # Destination: sibling of the thumbnails folder named "founded"
        small_dir = Path(self._small_dir.get())
        founded_dir = small_dir.parent / "founded"

        if not messagebox.askyesno(
                "Move matched thumbnails",
                f"Move {len(matched)} matched thumbnail(s) to:\n"
                f"  {founded_dir}\n\n"
                f"{len(kept)} unmatched will remain in the list.\n"
                f"(Original large files are NOT touched.)"):
            return

        founded_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        failed = 0
        for e in matched:
            dst = founded_dir / e.small_path.name
            # Avoid overwriting if name already exists
            if dst.exists():
                stem, sfx, ctr = dst.stem, dst.suffix, 1
                while dst.exists():
                    dst = founded_dir / f"{stem}_{ctr}{sfx}"
                    ctr += 1
            try:
                shutil.move(str(e.small_path), dst)
                moved += 1
            except Exception as ex:
                print(f"Move failed: {e.small_path} -> {ex}")
                failed += 1

        self.entries = kept
        self._filter.set("all")
        self._render_grid(self.entries)
        self._update_stats()
        msg = (f"Moved {moved} thumbnail(s) to founded/  ·  "
               f"{len(kept)} unmatched remain  ·  "
               f"Set a new Originals folder and click Run.")
        if failed:
            msg += f"  ({failed} failed to move)"
        self._set_status(msg)

    # ── filter ───────────────────────────────────────────────────────────────
    def _apply_filter(self):
        mode = self._filter.get()
        for i, entry in enumerate(self.entries):
            if i >= len(self._cards):
                break
            show = True
            if   mode == "p1"       and entry.phase != 1: show = False
            elif mode == "p2"       and entry.phase != 2: show = False
            elif mode == "p3"       and entry.phase != 3: show = False
            elif mode == "notfound" and entry.found:      show = False
            if show: self._cards[i].grid()
            else:    self._cards[i].grid_remove()

    # ── stats ─────────────────────────────────────────────────────────────────
    def _update_stats(self):
        e  = self.entries
        n  = len(e)
        p1 = sum(1 for x in e if x.phase == 1)
        p2 = sum(1 for x in e if x.phase == 2)
        p3 = sum(1 for x in e if x.phase == 3)
        nm = n - p1 - p2 - p3
        avg = (sum(x.match_score for x in e if x.found) / max(p1+p2+p3, 1))
        self._stats.config(text=(
            f"Total     : {n}\n"
            f"Phase 1   : {p1}  (exact)\n"
            f"Phase 2   : {p2}  (close)\n"
            f"Phase 3   : {p3}  (fuzzy)\n"
            f"Unmatched : {nm}\n"
            f"Avg score : {avg:.1f}%"
        ))

    # ── export txt ────────────────────────────────────────────────────────────
    def _export_txt(self):
        if not self.entries:
            messagebox.showinfo("Info", "Run matching first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile="image_match_results.txt")
        if not path:
            return

        lines = ["IMAGE MATCHER — TIERED RESULTS", "="*64, ""]
        for phase, label in [
            (1, "PHASE 1 — EXACT"),
            (2, "PHASE 2 — CLOSE"),
            (3, "PHASE 3 — FUZZY"),
        ]:
            group = [e for e in self.entries if e.phase == phase]
            lines += [f"[{label}]  ({len(group)} images)", ""]
            for e in group:
                lines += [
                    f"  Thumbnail : {e.small_path}",
                    f"  Original  : {e.match_path}",
                    f"  Score     : {e.match_score:.1f}%  (dist={e.match_dist})",
                    "",
                ]

        nm = [e for e in self.entries if not e.found]
        lines += [f"[NOT MATCHED]  ({len(nm)} images)", ""]
        for e in nm:
            lines.append(f"  {e.small_path}")

        found = len(self.entries) - len(nm)
        lines += [
            "", "="*64,
            f"Total: {len(self.entries)}  |  Matched: {found}  |  Unmatched: {len(nm)}",
        ]

        Path(path).write_text("\n".join(lines), encoding="utf-8")
        messagebox.showinfo("Saved", f"Results written to:\n{path}")

    # ── copy originals ────────────────────────────────────────────────────────
    def _copy_originals(self):
        dest = self._dest_dir.get()
        if not dest:
            dest = filedialog.askdirectory(title="Select destination")
            if not dest:
                return
            self._dest_dir.set(dest)

        dest_path = Path(dest)

        matched = [e for e in self.entries if e.found]
        if not matched:
            messagebox.showinfo("Info", "No matched originals to copy.")
            return

        # Sub-folders by phase
        phase_dirs = {
            1: dest_path / "Exact",
            2: dest_path / "Close",
            3: dest_path / "Fuzzy",
        }
        for d in phase_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        copied = skipped = 0
        counts = {1: 0, 2: 0, 3: 0}
        for e in matched:
            target_dir = phase_dirs[e.phase]
            dst = target_dir / e.match_path.name
            if dst.exists():
                stem, sfx, ctr = dst.stem, dst.suffix, 1
                while dst.exists():
                    dst = target_dir / f"{stem}_{ctr}{sfx}"
                    ctr += 1
            try:
                shutil.copy2(e.match_path, dst)
                copied += 1
                counts[e.phase] += 1
            except Exception as ex:
                print(f"Copy failed: {e.match_path} -> {ex}")
                skipped += 1

        msg = (
            f"Copied {copied} file(s) to:\n{dest_path}\n\n"
            f"  Exact\\  {counts[1]} file(s)\n"
            f"  Close\\  {counts[2]} file(s)\n"
            f"  Fuzzy\\  {counts[3]} file(s)"
        )
        if skipped:
            msg += f"\n\n({skipped} failed)"
        messagebox.showinfo("Done", msg)


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()
