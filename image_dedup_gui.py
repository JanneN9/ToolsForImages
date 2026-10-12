#!/usr/bin/env python3
"""
Duplicate Finder GUI  —  Recursive, Multi-Technique
----------------------------------------------------
Give it one Main folder; it scans every sub-folder for images and groups
together everything it thinks is a duplicate of something else — not just
byte-identical files (that's image_dedup.py), but near-duplicates and,
optionally, images that are "the same picture" semantically even though
every pixel differs (different render/re-save/crop).

TECHNIQUES (toggle each on/off, each has its own threshold slider):
  Hash match     : pHash + dHash, vectorized pairwise Hamming distance.
                   Distance 0   = EXACT  (byte-for-byte visually identical)
                   Distance 1-N = CLOSE  (same shot, re-saved/resized/cropped)
  Semantic match : CLIP embedding cosine similarity (optional, needs
                   torch + open_clip_torch). Finds "not so similar" images —
                   same scene/subject/composition rendered or shot
                   completely differently. This is what hash/pixel
                   comparison structurally cannot do (see image_matcher.py's
                   Phase 4 for the full explanation).

Duplicates can involve MORE than 2 files (e.g. the same photo re-saved
three times): matches are grouped transitively with union-find, so a
group shows every image that belonges together, side by side.

Each image in a group has its own Delete / Move buttons underneath it —
act on individual files without leaving the comparison view. Delete uses
the Recycle Bin via send2trash when available, otherwise asks for a very
explicit confirmation before permanently removing the file.

Requirements:
    pip install Pillow imagehash numpy
    pip install send2trash              # optional, Recycle-Bin delete instead of permanent
    pip install torch open_clip_torch   # optional, enables semantic matching

Works on Windows (incl. Anaconda/Miniconda), macOS and Linux — tkinter
ships with all three; no extra GUI toolkit needed.

Note on scale: matching is done with vectorized all-pairs comparisons
(one Hamming-distance / cosine-similarity matrix via numpy, not a Python
double loop), which is fast but holds an N×N matrix in memory. Tens of
thousands of images is fine; beyond ~50-100k you'll want to run it per
sub-folder instead of over everything at once.

Usage:
    python image_dedup_gui.py
"""

import os
import sys
import shutil
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# ── Windows HiDPI fix ────────────────────────────────────────────────────────
# Must run before the first Tk root is created anywhere in the process
# (including the temporary root _resolve_fonts() below creates at import
# time). See image_matcher.py for the full explanation.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    import imagehash
    import numpy as np
except ImportError:
    print("Missing dependencies. Run:  pip install Pillow imagehash numpy")
    sys.exit(1)

# send2trash is optional — Recycle-Bin delete instead of a permanent one
try:
    from send2trash import send2trash
    SEND2TRASH_AVAILABLE = True
except ImportError:
    SEND2TRASH_AVAILABLE = False

# torch + open_clip are optional — used for semantic (CLIP) matching
try:
    import torch
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# ── cross-platform font resolver ──────────────────────────────────────────
def _resolve_fonts():
    import tkinter.font as tkfont
    try:
        _root = tk.Tk()
        _root.withdraw()
        available = set(tkfont.families(_root))
        _root.destroy()
    except Exception:
        available = set()

    mono_candidates = [
        "Consolas", "DejaVu Sans Mono", "Liberation Mono", "Ubuntu Mono",
        "Noto Mono", "Noto Sans Mono", "FreeMono", "Andale Mono",
        "Nimbus Mono PS", "Courier New", "Courier",
    ]
    mono = next((f for f in mono_candidates if f in available), "Courier")

    return {
        "title":   (mono, 14, "bold"),
        "label":   (mono,  8),
        "label_b": (mono,  8, "bold"),
        "entry":   (mono,  9),
        "section": (mono,  9, "bold"),
        "badge":   (mono,  8, "bold"),
        "name":    (mono,  8),
        "info":    (mono,  8),
        "status":  (mono,  9),
        "btn":     (mono,  9),
        "btn_run": (mono, 11, "bold"),
    }

F = _resolve_fonts()


# ── colours & constants ────────────────────────────────────────────────────
BG_DARK   = "#0d0d1a"
BG_MID    = "#131325"
BG_CARD   = "#1a1a30"

C_GOLD    = "#ffd700"
C_GREEN   = "#00e676"
C_PURPLE  = "#b388ff"
C_RED     = "#ff1744"
C_BLUE    = "#448aff"
C_TEXT    = "#dde1f0"
C_DIM     = "#7986a8"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

THUMBNAIL_SIZE = (220, 220)
GRID_COLS      = 3

PREVIEW_THUMB_SIZE = (420, 420)   # double-click preview — bigger than the grid thumbnail
PREVIEW_COLS_SMALL = 3            # up to 9 images -> 3x3
PREVIEW_COLS_LARGE = 5            # more than 9 -> 5 wide (e.g. 5x2 for 10), scrolls beyond that

HASH_SIZE      = 8          # imagehash default -> 64-bit hash
EXACT_MAX      = 0          # Hamming distance: exact
CLOSE_MAX_DEF  = 6          # Hamming distance: close (slider-adjustable)
CLIP_MIN_SIM_DEF = 0.85     # cosine similarity (slider-adjustable)

PHASE_LABEL = {1: "EXACT", 2: "CLOSE", 3: "SEMANTIC"}
PHASE_COLOR = {1: C_GOLD, 2: C_GREEN, 3: C_PURPLE}


# ── data model ─────────────────────────────────────────────────────────────
@dataclass
class ImageInfo:
    path: Path
    size_bytes: int = 0
    width: int = 0
    height: int = 0


@dataclass
class DupGroup:
    gid: int
    images: List[ImageInfo]
    phase: int = 2            # best (lowest) technique number that linked the group
    best_score: float = 0.0

    @property
    def color(self):
        return PHASE_COLOR.get(self.phase, C_DIM)

    @property
    def label(self):
        return PHASE_LABEL.get(self.phase, "?")

    @property
    def reclaimable_bytes(self):
        """Size of every file in the group except the largest one kept."""
        sizes = sorted((im.size_bytes for im in self.images), reverse=True)
        return sum(sizes[1:])


# ── scanning helpers ─────────────────────────────────────────────────────────
def collect_images_recursive(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXT
    )


def hash_to_bits(hash_obj) -> np.ndarray:
    """imagehash hash -> flat 0/1 uint8 array (HASH_SIZE*HASH_SIZE bits)."""
    return hash_obj.hash.flatten().astype(np.uint8)


def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    """
    Pairwise Hamming distance for N binary vectors, vectorized.
    popcount(a XOR b) = popcount(a) + popcount(b) - 2*popcount(a AND b),
    and popcount(a AND b) for 0/1 vectors is just the dot product —
    so this turns into one BLAS matrix multiply instead of an O(n^2)
    Python loop.
    """
    bits_i = bits.astype(np.int32)
    ones = bits_i.sum(axis=1)
    overlap = bits_i @ bits_i.T
    return ones[:, None] + ones[None, :] - 2 * overlap


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


_clip_model_cache = {}


def clip_load():
    if not CLIP_AVAILABLE:
        return None, None, None
    if "model" in _clip_model_cache:
        c = _clip_model_cache
        return c["model"], c["preprocess"], c["device"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval().to(device)
    _clip_model_cache.update(model=model, preprocess=preprocess, device=device)
    return model, preprocess, device


def clip_embed(path: Path, model, preprocess, device) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()
    except Exception:
        return None


def find_duplicate_groups(paths: List[Path], use_hash: bool, use_clip: bool,
                           hash_close_max: int, clip_min_sim: float,
                           progress_cb=None) -> Tuple[List[List[int]], Dict[Tuple[int, int], dict]]:
    """
    Returns (groups, edge_info):
      groups    : list of index-lists (into `paths`), each a connected
                  component of size >= 2 — i.e. a duplicate group.
      edge_info : {(i, j): {"phase": int, "score": float}} for the
                  strongest technique that linked each matched pair,
                  used only to color/label groups in the UI.
    """
    n = len(paths)
    uf = UnionFind(n)
    edge_info: Dict[Tuple[int, int], dict] = {}

    def add_edge(i, j, phase, score):
        key = (min(i, j), max(i, j))
        uf.union(i, j)
        prev = edge_info.get(key)
        if prev is None or phase < prev["phase"]:
            edge_info[key] = {"phase": phase, "score": score}

    if use_hash:
        phash_bits = np.zeros((n, HASH_SIZE * HASH_SIZE), dtype=np.uint8)
        dhash_bits = np.zeros((n, HASH_SIZE * HASH_SIZE), dtype=np.uint8)
        valid = np.zeros(n, dtype=bool)
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    phash_bits[i] = hash_to_bits(imagehash.phash(img, hash_size=HASH_SIZE))
                    dhash_bits[i] = hash_to_bits(imagehash.dhash(img, hash_size=HASH_SIZE))
                valid[i] = True
            except Exception:
                pass
            if progress_cb and i % 20 == 0:
                progress_cb(i, n, "Hashing images")
        if progress_cb:
            progress_cb(n, n, "Hashing images")

        dist = np.minimum(hamming_matrix(phash_bits), hamming_matrix(dhash_bits))
        np.fill_diagonal(dist, 999)
        ii, jj = np.where(dist <= hash_close_max)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if i < j and valid[i] and valid[j]:
                d = int(dist[i, j])
                phase = 1 if d == EXACT_MAX else 2
                score = round(max(0.0, (1.0 - d / (HASH_SIZE * HASH_SIZE * 2))) * 100, 1)
                add_edge(i, j, phase, score)

    if use_clip and CLIP_AVAILABLE:
        model, preprocess, device = clip_load()
        embeddings: List[Optional[np.ndarray]] = [None] * n
        for i, p in enumerate(paths):
            embeddings[i] = clip_embed(p, model, preprocess, device)
            if progress_cb and i % 10 == 0:
                progress_cb(i, n, "Embedding images (CLIP)")
        if progress_cb:
            progress_cb(n, n, "Embedding images (CLIP)")

        dim = next((e.shape[0] for e in embeddings if e is not None), None)
        if dim is not None:
            embs = np.zeros((n, dim), dtype=np.float32)
            valid_c = np.zeros(n, dtype=bool)
            for i, e in enumerate(embeddings):
                if e is not None:
                    embs[i] = e
                    valid_c[i] = True
            sims = embs @ embs.T  # pre-normalized -> dot product == cosine
            np.fill_diagonal(sims, -1.0)
            ii, jj = np.where(sims >= clip_min_sim)
            for i, j in zip(ii.tolist(), jj.tolist()):
                if i < j and valid_c[i] and valid_c[j]:
                    add_edge(i, j, 3, round(float(sims[i, j]) * 100, 1))

    groups_map: Dict[int, List[int]] = {}
    touched = {i for pair in edge_info for i in pair}
    for i in touched:
        groups_map.setdefault(uf.find(i), []).append(i)
    groups = [sorted(idxs) for idxs in groups_map.values() if len(idxs) >= 2]
    return groups, edge_info


def build_dup_groups(paths: List[Path], groups_idx: List[List[int]],
                      edge_info: Dict[Tuple[int, int], dict]) -> List[DupGroup]:
    dup_groups = []
    for gid, idxs in enumerate(groups_idx, start=1):
        images = []
        for i in idxs:
            p = paths[i]
            try:
                size_bytes = p.stat().st_size
            except Exception:
                size_bytes = 0
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                w = h = 0
            images.append(ImageInfo(path=p, size_bytes=size_bytes, width=w, height=h))
        images.sort(key=lambda im: im.size_bytes, reverse=True)

        best_phase, best_score = 3, 0.0
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                key = (min(idxs[a], idxs[b]), max(idxs[a], idxs[b]))
                info = edge_info.get(key)
                if info:
                    best_phase = min(best_phase, info["phase"])
                    best_score = max(best_score, info["score"])

        dup_groups.append(DupGroup(gid=gid, images=images, phase=best_phase, best_score=best_score))
    return dup_groups


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


# ── GUI ─────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Duplicate Finder  —  Recursive  (Hash+CLIP)")
        self.geometry("1280x860")
        self.configure(bg=BG_DARK)
        self.minsize(960, 640)

        self.groups: List[DupGroup] = []
        self.selected_group: Optional[DupGroup] = None
        self._photos = []

        self._main_dir = tk.StringVar()
        self._move_dir = tk.StringVar()
        self._use_hash = tk.BooleanVar(value=True)
        self._use_clip = tk.BooleanVar(value=False)
        self._hash_max = tk.IntVar(value=CLOSE_MAX_DEF)
        self._clip_min_sim = tk.DoubleVar(value=CLIP_MIN_SIM_DEF)

        self._status_var = tk.StringVar(value="Pick a Main folder and click Scan.")
        self._prog_var = tk.DoubleVar()
        self._prog_lbl = tk.StringVar(value="")

        self._build_ui()

    # ── build UI ────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_toolbar()

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg=BG_MID, width=300)
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

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=BG_MID, pady=10)
        bar.pack(fill=tk.X)

        tk.Label(bar, text="◈  DUPLICATE FINDER", bg=BG_MID,
                 fg=C_BLUE, font=F["title"], padx=14).pack(side=tk.LEFT)

        f = tk.Frame(bar, bg=BG_MID)
        f.pack(side=tk.LEFT, padx=8)
        tk.Label(f, text="📁 Main folder", bg=BG_MID, fg=C_DIM, font=F["label"]).pack(anchor=tk.W)
        row = tk.Frame(f, bg=BG_MID)
        row.pack()
        tk.Entry(row, textvariable=self._main_dir, width=28,
                 bg="#1c1c38", fg=C_TEXT, relief=tk.FLAT,
                 insertbackground=C_TEXT, font=F["entry"]).pack(side=tk.LEFT)
        tk.Button(row, text="…", command=lambda: self._browse(self._main_dir),
                  bg="#28284a", fg=C_TEXT, relief=tk.FLAT, padx=5,
                  cursor="hand2").pack(side=tk.LEFT, padx=1)

        f2 = tk.Frame(bar, bg=BG_MID)
        f2.pack(side=tk.LEFT, padx=8)
        tk.Label(f2, text="📂 Move-to folder", bg=BG_MID, fg=C_DIM, font=F["label"]).pack(anchor=tk.W)
        row2 = tk.Frame(f2, bg=BG_MID)
        row2.pack()
        tk.Entry(row2, textvariable=self._move_dir, width=22,
                 bg="#1c1c38", fg=C_TEXT, relief=tk.FLAT,
                 insertbackground=C_TEXT, font=F["entry"]).pack(side=tk.LEFT)
        tk.Button(row2, text="…", command=lambda: self._browse(self._move_dir),
                  bg="#28284a", fg=C_TEXT, relief=tk.FLAT, padx=5,
                  cursor="hand2").pack(side=tk.LEFT, padx=1)

        tf = tk.Frame(bar, bg=BG_MID)
        tf.pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(tf, text="Hash (Exact+Close)", variable=self._use_hash,
                        bg=BG_MID, fg=C_GOLD, selectcolor=BG_DARK,
                        activebackground=BG_MID, font=F["label_b"]).grid(row=0, column=0, sticky=tk.W)
        tk.Label(tf, text="max dist:", bg=BG_MID, fg=C_DIM, font=F["label"]).grid(row=0, column=1, padx=(8, 2))
        tk.Scale(tf, from_=0, to=20, orient=tk.HORIZONTAL, variable=self._hash_max,
                 bg=BG_MID, fg=C_TEXT, troughcolor="#222244",
                 highlightthickness=0, length=100, showvalue=True).grid(row=0, column=2)

        tk.Checkbutton(tf, text="Semantic (CLIP)", variable=self._use_clip,
                        bg=BG_MID, fg=C_PURPLE, selectcolor=BG_DARK,
                        activebackground=BG_MID, font=F["label_b"]).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        tk.Label(tf, text="min sim:", bg=BG_MID, fg=C_DIM, font=F["label"]).grid(row=1, column=1, padx=(8, 2), pady=(4, 0))
        tk.Scale(tf, from_=0.50, to=0.99, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=self._clip_min_sim, bg=BG_MID, fg=C_TEXT, troughcolor="#222244",
                 highlightthickness=0, length=100, showvalue=True).grid(row=1, column=2, pady=(4, 0))

        tk.Button(bar, text="▶  SCAN", command=self._start_scan,
                  bg=C_BLUE, fg=BG_DARK, font=F["btn_run"],
                  relief=tk.FLAT, padx=18, pady=7, cursor="hand2").pack(side=tk.RIGHT, padx=16)

        clip_text = "CLIP ✓" if CLIP_AVAILABLE else "CLIP ✗ (pip install torch open_clip_torch)"
        tk.Label(bar, text=clip_text, bg=BG_MID,
                 fg=C_GREEN if CLIP_AVAILABLE else C_RED, font=F["label"]).pack(side=tk.RIGHT, padx=6)
        trash_text = "Recycle Bin ✓" if SEND2TRASH_AVAILABLE else "Recycle Bin ✗ (pip install send2trash)"
        tk.Label(bar, text=trash_text, bg=BG_MID,
                 fg=C_GREEN if SEND2TRASH_AVAILABLE else C_RED, font=F["label"]).pack(side=tk.RIGHT, padx=6)

    def _build_sidebar(self, parent):
        def section(text):
            tk.Label(parent, text=text, bg=BG_MID, fg=C_BLUE, font=F["section"],
                      pady=6, padx=12, anchor=tk.W).pack(fill=tk.X)

        section("GROUPS")
        list_frame = tk.Frame(parent, bg=BG_MID)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8)
        sb = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._group_list = tk.Listbox(
            list_frame, bg="#0d0d1e", fg=C_TEXT, selectbackground="#2a2a50",
            relief=tk.FLAT, font=F["entry"], yscrollcommand=sb.set,
            highlightthickness=0, activestyle="none")
        sb.config(command=self._group_list.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._group_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._group_list.bind("<<ListboxSelect>>", self._on_group_select)

        section("STATS")
        self._stats = tk.Label(parent, text="—", bg=BG_MID, fg=C_DIM,
                                font=F["info"], justify=tk.LEFT, padx=14)
        self._stats.pack(anchor=tk.W, pady=(0, 10))

        section("LEGEND")
        for col, lbl in [
            (C_GOLD,   "Exact     (hash dist 0)"),
            (C_GREEN,  "Close     (hash dist ≤ N)"),
            (C_PURPLE, "Semantic  (CLIP cosine)"),
        ]:
            row = tk.Frame(parent, bg=BG_MID)
            row.pack(anchor=tk.W, padx=12, pady=1)
            tk.Label(row, text="█", fg=col, bg=BG_MID, font=F["btn"]).pack(side=tk.LEFT)
            tk.Label(row, text=lbl, fg=C_DIM, bg=BG_MID, font=F["label"]).pack(side=tk.LEFT, padx=4)

    def _build_grid_area(self, parent):
        ph = tk.Frame(parent, bg=BG_DARK)
        ph.pack(fill=tk.X, padx=4, pady=(4, 0))
        tk.Label(ph, textvariable=self._prog_lbl, bg=BG_DARK, fg=C_DIM, font=F["info"]).pack(side=tk.LEFT, padx=4)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Horizontal.TProgressbar", background=C_BLUE, troughcolor="#1c1c38", borderwidth=0)
        ttk.Progressbar(ph, variable=self._prog_var, maximum=100, length=300).pack(side=tk.LEFT)

        self._group_title = tk.Label(parent, text="Select a group on the left",
                                      bg=BG_DARK, fg=C_TEXT, font=F["section"], anchor=tk.W, padx=8)
        self._group_title.pack(fill=tk.X, pady=(4, 0))

        container = tk.Frame(parent, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(container, bg=BG_DARK, highlightthickness=0)
        vsb = tk.Scrollbar(container, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._gframe = tk.Frame(self._canvas, bg=BG_DARK)
        self._cwin = self._canvas.create_window((0, 0), window=self._gframe, anchor=tk.NW)
        self._gframe.bind("<Configure>", lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfig(self._cwin, width=e.width))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._canvas.bind_all(seq, self._scroll)

    def _scroll(self, event):
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _browse(self, var):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    # ── scan ─────────────────────────────────────────────────────────────────
    def _start_scan(self):
        main_dir = Path(self._main_dir.get())
        if not main_dir.is_dir():
            messagebox.showerror("Error", "Main folder not found.")
            return
        if not self._use_hash.get() and not self._use_clip.get():
            messagebox.showwarning("No technique selected", "Enable Hash and/or Semantic matching.")
            return
        if self._use_clip.get() and not CLIP_AVAILABLE:
            messagebox.showwarning(
                "CLIP not installed",
                "Semantic matching needs torch + open_clip_torch.\n"
                "Install with:  pip install torch open_clip_torch\n\n"
                "Continuing with Hash matching only.")

        self.groups = []
        self.selected_group = None
        self._group_list.delete(0, tk.END)
        self._clear_grid()
        self._group_title.config(text="Scanning…")
        self._prog_var.set(0)

        use_hash = self._use_hash.get()
        use_clip = self._use_clip.get() and CLIP_AVAILABLE
        hash_max = self._hash_max.get()
        clip_min_sim = self._clip_min_sim.get()

        def worker():
            self.after(0, lambda: self._status_var.set("Scanning folders…"))
            paths = collect_images_recursive(main_dir)
            if not paths:
                self.after(0, lambda: messagebox.showinfo("Info", "No images found."))
                return
            self.after(0, lambda: self._status_var.set(f"{len(paths)} images found — matching…"))

            def prog(cur, total, label):
                pct = (cur / max(total, 1)) * 100
                self.after(0, lambda: self._prog_var.set(pct))
                self.after(0, lambda: self._prog_lbl.set(f"{label}  {cur}/{total}"))

            groups_idx, edge_info = find_duplicate_groups(
                paths, use_hash, use_clip, hash_max, clip_min_sim, prog)
            dup_groups = build_dup_groups(paths, groups_idx, edge_info)
            self.after(0, lambda: self._scan_done(len(paths), dup_groups))

        threading.Thread(target=worker, daemon=True).start()

    def _scan_done(self, n_scanned, dup_groups):
        self.groups = dup_groups
        self._prog_var.set(100)
        self._prog_lbl.set("Done")

        for g in self.groups:
            self._group_list.insert(tk.END, f"Group {g.gid}  ({len(g.images)} imgs)  [{g.label}]")
            self._group_list.itemconfig(tk.END, fg=g.color)

        total_dupes = sum(len(g.images) - 1 for g in self.groups)
        reclaimable = sum(g.reclaimable_bytes for g in self.groups)
        self._stats.config(text=(
            f"Scanned    : {n_scanned}\n"
            f"Groups     : {len(self.groups)}\n"
            f"Extra files: {total_dupes}\n"
            f"Reclaimable: {human_size(reclaimable)}"
        ))
        self._status_var.set(f"Done — {len(self.groups)} duplicate group(s) found among {n_scanned} images.")
        if self.groups:
            self._group_list.selection_set(0)
            self._show_group(self.groups[0])
        else:
            self._group_title.config(text="No duplicates found")

    # ── group selection & grid ──────────────────────────────────────────────
    def _on_group_select(self, _evt):
        sel = self._group_list.curselection()
        if not sel:
            return
        self._show_group(self.groups[sel[0]])

    def _show_group(self, group: DupGroup):
        self.selected_group = group
        self._group_title.config(
            text=f"Group {group.gid}  —  {len(group.images)} images  —  "
                 f"{group.label} ({group.best_score:.0f}%)",
            fg=group.color)
        self._render_group(group)

    def _clear_grid(self):
        for w in self._gframe.winfo_children():
            w.destroy()
        self._photos.clear()

    def _render_group(self, group: DupGroup):
        self._clear_grid()
        for idx, info in enumerate(group.images):
            r, c = divmod(idx, GRID_COLS)
            card = self._make_card(group, info)
            card.grid(row=r, column=c, padx=8, pady=8, sticky=tk.NSEW)
        self._canvas.yview_moveto(0)

    def _make_card(self, group: DupGroup, info: ImageInfo):
        card = tk.Frame(self._gframe, bg=BG_CARD,
                         highlightbackground=group.color, highlightthickness=2)

        try:
            with Image.open(info.path) as raw:
                img = raw.convert("RGB")
            img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
            canvas_img = Image.new("RGB", THUMBNAIL_SIZE, (26, 26, 48))
            off = ((THUMBNAIL_SIZE[0] - img.width) // 2, (THUMBNAIL_SIZE[1] - img.height) // 2)
            canvas_img.paste(img, off)
            photo = ImageTk.PhotoImage(canvas_img)
        except Exception:
            photo = ImageTk.PhotoImage(Image.new("RGB", THUMBNAIL_SIZE, (30, 30, 50)))
        self._photos.append(photo)

        thumb_lbl = tk.Label(card, image=photo, bg=BG_CARD, cursor="hand2")
        thumb_lbl.pack(padx=6, pady=(6, 2))
        for w in (card, thumb_lbl):
            w.bind("<Double-Button-1>", lambda _e, g=group: self._open_preview(g))

        name = info.path.name
        if len(name) > 30:
            name = name[:27] + "…"
        tk.Label(card, text=name, bg=BG_CARD, fg=C_TEXT, font=F["name"]).pack()
        tk.Label(card, text=str(info.path.parent), bg=BG_CARD, fg=C_DIM,
                 font=F["label"], wraplength=THUMBNAIL_SIZE[0]).pack()
        tk.Label(card, text=f"{info.width}×{info.height}  ·  {human_size(info.size_bytes)}",
                 bg=BG_CARD, fg=C_DIM, font=F["label"]).pack(pady=(0, 6))

        btn_row = tk.Frame(card, bg=BG_CARD)
        btn_row.pack(pady=(0, 8))
        tk.Button(btn_row, text="📁 Move", command=lambda: self._move_image(group, info),
                  bg="#1c1c38", fg=C_TEXT, relief=tk.FLAT, font=F["btn"],
                  padx=8, cursor="hand2").pack(side=tk.LEFT, padx=3)
        tk.Button(btn_row, text="🗑 Delete", command=lambda: self._delete_image(group, info),
                  bg="#2a1a1a", fg=C_RED, relief=tk.FLAT, font=F["btn"],
                  padx=8, cursor="hand2").pack(side=tk.LEFT, padx=3)

        return card

    # ── full-size preview (double-click) ────────────────────────────────────
    def _open_preview(self, group: DupGroup):
        """Double-click a thumbnail: show every image in the group at a much
        larger size, side by side, so differences are actually visible. Grid
        is capped at 3 wide for up to 9 images (3x3) or 5 wide beyond that
        (e.g. 5x2 for 10) — any more than that just scrolls."""
        win = tk.Toplevel(self)
        win.title(f"Group {group.gid}  —  Preview ({len(group.images)} images)")
        win.configure(bg=BG_DARK)
        win.geometry("1200x800")

        container = tk.Frame(win, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, bg=BG_DARK, highlightthickness=0)
        vsb = tk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        hsb = tk.Scrollbar(container, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        canvas.grid(row=0, column=0, sticky=tk.NSEW)
        vsb.grid(row=0, column=1, sticky=tk.NS)
        hsb.grid(row=1, column=0, sticky=tk.EW)

        gframe = tk.Frame(canvas, bg=BG_DARK)
        canvas.create_window((0, 0), window=gframe, anchor=tk.NW)
        # Deliberately NOT forcing gframe's width to the canvas width here
        # (unlike the main grid) — the whole point is that when the window
        # is shrunk (e.g. to 880x660) and the cards don't fit, the content
        # keeps its natural size and a horizontal scrollbar appears instead
        # of cards getting squeezed.
        gframe.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))

        def scroll_y(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def scroll_x(event):
            if event.num == 6:
                canvas.xview_scroll(-1, "units")
            elif event.num == 7:
                canvas.xview_scroll(1, "units")
            else:
                canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        def restore_main_scroll_and_close():
            # bind_all is process-wide, so give mouse-wheel scrolling back to
            # the main window's canvas once this preview closes.
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self._canvas.bind_all(seq, self._scroll)
            for seq in ("<Shift-MouseWheel>", "<Shift-Button-4>", "<Shift-Button-5>",
                        "<Button-6>", "<Button-7>"):
                try:
                    self._canvas.unbind_all(seq)
                except tk.TclError:
                    pass
            win.destroy()

        def safe_bind_all(seq, handler):
            # Not every Tk build recognizes every sequence (e.g. Button-6/7
            # for a horizontal wheel raise "bad button number" on some
            # platforms) — skip whatever isn't supported instead of crashing.
            try:
                canvas.bind_all(seq, handler)
            except tk.TclError:
                pass

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            safe_bind_all(seq, scroll_y)
        # Horizontal: Shift+wheel on Windows/Mac, Shift+Button-4/5 on some X11
        # setups, and dedicated Button-6/7 on others with a horizontal wheel.
        for seq in ("<Shift-MouseWheel>", "<Shift-Button-4>", "<Shift-Button-5>",
                    "<Button-6>", "<Button-7>"):
            safe_bind_all(seq, scroll_x)
        win.protocol("WM_DELETE_WINDOW", restore_main_scroll_and_close)

        photos = []
        win._photos = photos  # keep references alive for the window's lifetime

        cols = PREVIEW_COLS_SMALL if len(group.images) <= 9 else PREVIEW_COLS_LARGE
        for idx, info in enumerate(group.images):
            r, c = divmod(idx, cols)
            card = tk.Frame(gframe, bg=BG_CARD, highlightbackground=group.color, highlightthickness=2)
            card.grid(row=r, column=c, padx=8, pady=8, sticky=tk.NSEW)

            try:
                with Image.open(info.path) as raw:
                    img = raw.convert("RGB")
                img.thumbnail(PREVIEW_THUMB_SIZE, Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
            except Exception:
                photo = ImageTk.PhotoImage(Image.new("RGB", PREVIEW_THUMB_SIZE, (30, 30, 50)))
            photos.append(photo)

            tk.Label(card, image=photo, bg=BG_CARD).pack(padx=6, pady=(6, 2))
            tk.Label(card, text=info.path.name, bg=BG_CARD, fg=C_TEXT, font=F["name"]).pack()
            tk.Label(card, text=str(info.path.parent), bg=BG_CARD, fg=C_DIM,
                      font=F["label"], wraplength=PREVIEW_THUMB_SIZE[0]).pack()
            tk.Label(card, text=f"{info.width}×{info.height}  ·  {human_size(info.size_bytes)}",
                      bg=BG_CARD, fg=C_DIM, font=F["label"]).pack(pady=(0, 6))

        tk.Button(win, text="Close", command=restore_main_scroll_and_close,
                   bg="#1c1c38", fg=C_TEXT, relief=tk.FLAT, font=F["btn"],
                   padx=12, pady=4, cursor="hand2").pack(pady=8)

    # ── per-image actions ───────────────────────────────────────────────────
    def _remove_from_group(self, group: DupGroup, info: ImageInfo):
        group.images.remove(info)
        if len(group.images) < 2:
            # No longer a duplicate group — drop it entirely.
            idx = self.groups.index(group)
            self.groups.pop(idx)
            self._group_list.delete(idx)
            if self.groups:
                next_idx = min(idx, len(self.groups) - 1)
                self._group_list.selection_set(next_idx)
                self._show_group(self.groups[next_idx])
            else:
                self.selected_group = None
                self._clear_grid()
                self._group_title.config(text="No duplicates left", fg=C_TEXT)
        else:
            self._render_group(group)

    def _move_image(self, group: DupGroup, info: ImageInfo):
        dest = self._move_dir.get()
        if not dest:
            dest = filedialog.askdirectory(title="Select destination")
            if not dest:
                return
            self._move_dir.set(dest)
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)

        dst = dest_path / info.path.name
        if dst.exists():
            stem, sfx, ctr = dst.stem, dst.suffix, 1
            while dst.exists():
                dst = dest_path / f"{stem}_{ctr}{sfx}"
                ctr += 1
        try:
            shutil.move(str(info.path), dst)
        except Exception as ex:
            messagebox.showerror("Move failed", f"{info.path}\n\n{ex}")
            return
        self._status_var.set(f"Moved {info.path.name} -> {dst}")
        self._remove_from_group(group, info)

    def _delete_image(self, group: DupGroup, info: ImageInfo):
        if SEND2TRASH_AVAILABLE:
            if not messagebox.askyesno("Delete", f"Move this file to the Recycle Bin?\n\n{info.path}"):
                return
            try:
                send2trash(str(info.path))
            except Exception as ex:
                messagebox.showerror("Delete failed", f"{info.path}\n\n{ex}")
                return
        else:
            if not messagebox.askyesno(
                    "Permanently delete?",
                    f"send2trash is not installed, so this CANNOT be undone.\n\n"
                    f"Permanently delete:\n{info.path}\n\n"
                    f"(pip install send2trash to use the Recycle Bin instead.)"):
                return
            try:
                info.path.unlink()
            except Exception as ex:
                messagebox.showerror("Delete failed", f"{info.path}\n\n{ex}")
                return
        self._status_var.set(f"Deleted {info.path.name}")
        self._remove_from_group(group, info)


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()

    try:
        fontsystem = app.tk.call("tk", "pkgconfig", "get", "fontsystem")
    except Exception:
        fontsystem = "unknown"
    print("── image_dedup_gui font diagnostics ─────────────────────────────")
    print(f"  Python executable : {sys.executable}")
    print(f"  Tk font system    : {fontsystem}   (want: xft on Linux)")
    print("───────────────────────────────────────────────────────────────")

    app.mainloop()
