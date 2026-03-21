#!/usr/bin/env python3
"""
Image Sorter GUI
----------------
Browse a folder of generated images (paired *Own*.png / *Pub*.png files)
and quickly sort them with one-click action buttons.

Naming convention
  *Own*.png  — the "own/private" version  (higher quality / unprocessed)
  *Pub*.png  — the "published" version    (posted online)

Pairing strategy
  Files are paired by perceptual hash (pHash) similarity, NOT by filename
  numbers.  This handles cases where Own/Pub counters have drifted apart.
  A threshold slider in the toolbar controls how loose the matching is.
  Unmatched files are shown at the end with a warning border.

Actions (buttons on every card)
  [Save]    — move Own  → Save/,    delete Pub
  [Civit]   — move Pub  → Civit/,   also move Own → Save/
  [Improve] — move Own  → Improve/, delete Pub
  [Maybe]   — move Own  → Maybe/,   delete Pub
  [No]      — delete both files

Sub-folders are created automatically under the source folder.

Requirements:
    pip install Pillow imagehash
"""

import sys
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Missing dependency.  Run:  pip install Pillow imagehash")
    sys.exit(1)

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

# ── constants / colours (same palette as image_matcher) ─────────────────────
THUMBNAIL_SIZE = (160, 160)
GRID_COLS      = 5

BG_DARK  = "#0d0d1a"
BG_MID   = "#131325"
BG_CARD  = "#1a1a30"

C_GOLD   = "#ffd700"
C_GREEN  = "#00e676"
C_TEAL   = "#26c6da"
C_RED    = "#ff1744"
C_BLUE   = "#448aff"
C_ORANGE = "#ff9100"
C_PURPLE = "#ce93d8"
C_TEXT   = "#dde1f0"
C_DIM    = "#7986a8"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

# Button definitions: (label, colour, folder_or_None)
# folder=None means "delete"
ACTIONS = [
    ("Save",    C_GREEN,  "Save"),
    ("Civit",   C_GOLD,   "Civit"),
    ("Improve", C_TEAL,   "Improve"),
    ("Maybe",   C_ORANGE, "Maybe"),
    ("No",      C_RED,    None),
]

SUB_FOLDERS = ["Save", "Civit", "Improve", "Maybe"]


# ── pair discovery ────────────────────────────────────────────────────────────
def _phash(path: Path):
    """Compute pHash for one image; returns None on failure."""
    try:
        with Image.open(path) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception:
        return None


def _fallback_key(path: Path) -> str:
    """Strip Own/Pub tokens and trailing numbers to get a base key."""
    import re
    stem = path.stem
    for tok in ("OwnA", "PubA", "Own", "Pub"):
        stem = stem.replace(tok, "")
    stem = re.sub(r"[_\-]\d+$", "", stem).strip("_- ")
    return stem


def find_pairs(folder: Path, max_dist: int = 10, progress_cb=None):
    """
    Match Own* files to Pub* files using pHash nearest-neighbour.

    Algorithm
    ---------
    1. Separate folder contents into own_list / pub_list by filename token.
    2. Compute pHash for every file (fast — ~5 ms/image).
    3. For each Own, find the closest Pub by hash distance (greedy).
       If distance <= max_dist  → paired.
       Otherwise Own is listed without a partner.
    4. Any Pub not claimed by an Own is listed as a solo Pub entry.

    Returns list of (own_path_or_None, pub_path_or_None, label_str, dist_int)
    """
    own_list: list[Path] = []
    pub_list: list[Path] = []

    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
            continue
        stem = p.stem
        if "Own" in stem:
            own_list.append(p)
        elif "Pub" in stem:
            pub_list.append(p)

    total = len(own_list) + len(pub_list)
    done  = [0]

    def tick(label=""):
        done[0] += 1
        if progress_cb:
            progress_cb(done[0], total, label)

    # ── hash own files ───────────────────────────────────────────────
    if IMAGEHASH_AVAILABLE:
        own_hashes = []
        for p in own_list:
            own_hashes.append((p, _phash(p)))
            tick("Hashing Own files")

        pub_hashes = []
        for p in pub_list:
            pub_hashes.append((p, _phash(p)))
            tick("Hashing Pub files")

        # ── greedy nearest-neighbour matching ────────────────────────
        available_pub = list(pub_hashes)  # shrinks as pubs are claimed
        pairs = []

        for own_path, own_h in own_hashes:
            best_dist = 999
            best_idx  = -1
            if own_h is not None:
                for i, (pub_path, pub_h) in enumerate(available_pub):
                    if pub_h is None:
                        continue
                    d = own_h - pub_h
                    if d < best_dist:
                        best_dist = d
                        best_idx  = i

            if best_idx >= 0 and best_dist <= max_dist:
                pub_path, _ = available_pub.pop(best_idx)
                label = _fallback_key(own_path) or own_path.stem
                pairs.append((own_path, pub_path, label, best_dist))
            else:
                label = _fallback_key(own_path) or own_path.stem
                pairs.append((own_path, None, label, 999))

        # leftover pubs with no own partner
        for pub_path, _ in available_pub:
            label = _fallback_key(pub_path) or pub_path.stem
            pairs.append((None, pub_path, label, 999))

    else:
        # ── fallback: stem-based matching (original behaviour) ───────
        own_map: dict[str, Path] = {}
        pub_map: dict[str, Path] = {}
        for p in own_list:
            own_map[_fallback_key(p)] = p
        for p in pub_list:
            pub_map[_fallback_key(p)] = p
        all_keys = sorted(set(own_map) | set(pub_map))
        pairs = [(own_map.get(k), pub_map.get(k), k, 0) for k in all_keys]

    return pairs


# ── GUI ──────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Sorter  —  Own / Pub")
        self.geometry("1280x860")
        self.configure(bg=BG_DARK)
        self.minsize(900, 600)

        self._folder: Optional[Path] = None
        self._pairs   = []          # list of (own, pub, label, dist)
        self._photos  = []          # keep PhotoImage refs alive
        self._cards   = []          # card frames
        self._done_count = 0

        self._folder_var  = tk.StringVar()
        self._status_var  = tk.StringVar(value="Open a folder to begin.")
        self._prog_var    = tk.DoubleVar(value=0)
        self._prog_lbl    = tk.StringVar(value="")
        self._threshold   = tk.IntVar(value=10)
        self._folder_var.trace_add("write", self._on_folder_changed)
        self._after_id = None

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_toolbar()
        self._build_grid_area()
        tk.Label(self, textvariable=self._status_var,
                 bg=BG_MID, fg=C_DIM, anchor=tk.W,
                 padx=12, pady=3,
                 font=("Consolas", 9)).pack(fill=tk.X, side=tk.BOTTOM)

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=BG_MID, pady=8)
        bar.pack(fill=tk.X)

        tk.Label(bar, text="⬡  IMAGE SORTER", bg=BG_MID, fg=C_BLUE,
                 font=("Courier New", 14, "bold"), padx=14).pack(side=tk.LEFT)

        # Folder picker
        ff = tk.Frame(bar, bg=BG_MID)
        ff.pack(side=tk.LEFT, padx=10)
        tk.Label(ff, text="📁 Source folder", bg=BG_MID,
                 fg=C_DIM, font=("Courier New", 8)).pack(anchor=tk.W)
        row = tk.Frame(ff, bg=BG_MID)
        row.pack()
        self._folder_entry = tk.Entry(
            row, textvariable=self._folder_var, width=40,
            bg="#1c1c38", fg=C_TEXT, relief=tk.FLAT,
            insertbackground=C_TEXT, font=("Consolas", 9))
        self._folder_entry.pack(side=tk.LEFT)
        tk.Button(row, text="…", command=self._browse,
                  bg="#28284a", fg=C_TEXT, relief=tk.FLAT,
                  padx=6, cursor="hand2").pack(side=tk.LEFT, padx=2)

        # pHash threshold slider
        if IMAGEHASH_AVAILABLE:
            sf = tk.Frame(bar, bg=BG_MID)
            sf.pack(side=tk.LEFT, padx=10)
            tk.Label(sf, text="pHash threshold", bg=BG_MID,
                     fg=C_DIM, font=("Courier New", 8)).pack(anchor=tk.W)
            srow = tk.Frame(sf, bg=BG_MID)
            srow.pack()
            self._thr_lbl = tk.Label(srow, textvariable=self._threshold,
                     bg=BG_MID, fg=C_TEAL,
                     font=("Courier New", 9, "bold"), width=3)
            self._thr_lbl.pack(side=tk.LEFT)
            tk.Scale(srow, from_=1, to=30, orient=tk.HORIZONTAL,
                     variable=self._threshold,
                     bg=BG_MID, fg=C_TEXT, troughcolor="#222244",
                     highlightthickness=0, length=100, showvalue=False
                     ).pack(side=tk.LEFT)
            tk.Button(srow, text="Re-match", command=self._rematch,
                      bg="#28284a", fg=C_TEAL, relief=tk.FLAT,
                      font=("Courier New", 8), padx=6, cursor="hand2"
                      ).pack(side=tk.LEFT, padx=4)

        # Progress bar
        pf = tk.Frame(bar, bg=BG_MID)
        pf.pack(side=tk.LEFT, padx=8)
        tk.Label(pf, textvariable=self._prog_lbl,
                 bg=BG_MID, fg=C_DIM,
                 font=("Consolas", 8)).pack(anchor=tk.W)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Horizontal.TProgressbar",
                        background=C_BLUE, troughcolor="#1c1c38", borderwidth=0)
        ttk.Progressbar(pf, variable=self._prog_var,
                        maximum=100, length=180).pack()

        # imagehash availability indicator
        ih_text  = "imagehash ✓" if IMAGEHASH_AVAILABLE else "imagehash ✗ (pip install imagehash)"
        ih_color = C_GREEN if IMAGEHASH_AVAILABLE else C_RED
        tk.Label(bar, text=ih_text, bg=BG_MID, fg=ih_color,
                 font=("Courier New", 8)).pack(side=tk.RIGHT, padx=8)

        # Legend
        legend = tk.Frame(bar, bg=BG_MID)
        legend.pack(side=tk.RIGHT, padx=10)
        tk.Label(legend, text="Actions:", bg=BG_MID, fg=C_DIM,
                 font=("Courier New", 8)).pack(anchor=tk.W)
        leg_row = tk.Frame(legend, bg=BG_MID)
        leg_row.pack()
        for label, colour, _ in ACTIONS:
            tk.Label(leg_row, text=f"[{label}]", bg=BG_MID, fg=colour,
                     font=("Courier New", 8, "bold"), padx=4).pack(side=tk.LEFT)

        # Counter
        self._counter_var = tk.StringVar(value="0 pairs")
        tk.Label(bar, textvariable=self._counter_var,
                 bg=BG_MID, fg=C_DIM,
                 font=("Consolas", 9)).pack(side=tk.RIGHT, padx=16)

    def _build_grid_area(self):
        container = tk.Frame(self, bg=BG_DARK)
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
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _browse(self):
        p = filedialog.askdirectory(title="Select source folder")
        if p:
            self._folder_var.set(p)

    def _on_folder_changed(self, *_):
        if self._after_id:
            self.after_cancel(self._after_id)
        def check():
            self._after_id = None
            p = Path(self._folder_var.get())
            if p.is_dir():
                self._load_folder(p)
        self._after_id = self.after(400, check)

    def _rematch(self):
        """Re-run pairing with updated threshold (only unprocessed pairs)."""
        if self._folder and self._folder.is_dir():
            self._load_folder(self._folder)

    def _load_folder(self, folder: Path):
        self._folder = folder
        self._done_count = 0
        self._prog_var.set(0)
        self._prog_lbl.set("")
        self._status_var.set(f"Hashing images in {folder.name} …")
        threshold = self._threshold.get()

        def prog(cur, total, label):
            pct = (cur / max(total, 1)) * 100
            self.after(0, lambda: self._prog_var.set(pct))
            self.after(0, lambda: self._prog_lbl.set(f"{label}  {cur}/{total}"))

        def worker():
            pairs = find_pairs(folder, max_dist=threshold, progress_cb=prog)
            self.after(0, lambda: self._render_pairs(pairs))

        threading.Thread(target=worker, daemon=True).start()

    def _render_pairs(self, pairs):
        for w in self._gframe.winfo_children():
            w.destroy()
        self._photos.clear()
        self._cards.clear()
        self._pairs = pairs
        self._prog_var.set(100)
        self._prog_lbl.set("")

        matched   = sum(1 for o, p, _, d in pairs if o and p)
        unmatched = len(pairs) - matched
        self._counter_var.set(f"{matched} pairs  +  {unmatched} solo")
        self._status_var.set(
            f"{matched} matched pair(s),  {unmatched} unmatched  —  "
            f"click an action button to sort.")

        for idx, (own, pub, label, dist) in enumerate(pairs):
            r, c = divmod(idx, GRID_COLS)
            card = self._make_card(own, pub, label, dist, idx)
            card.grid(row=r, column=c, padx=5, pady=5, sticky=tk.NSEW)
            self._cards.append(card)

        self._canvas.yview_moveto(0)

    # ── card builder ─────────────────────────────────────────────────────────
    def _make_card(self, own: Optional[Path], pub: Optional[Path],
                   label: str, dist: int, idx: int) -> tk.Frame:

        paired = own is not None and pub is not None
        border_col = C_BLUE if paired else C_RED

        card = tk.Frame(self._gframe, bg=BG_CARD,
                        highlightbackground=border_col,
                        highlightthickness=2, cursor="arrow")

        # Thumbnail — prefer Own, fall back to Pub
        display_path = own or pub
        photo = self._load_thumb(display_path)
        self._photos.append(photo)

        thumb_lbl = tk.Label(card, image=photo, bg=BG_CARD, cursor="hand2")
        thumb_lbl.pack()
        thumb_lbl.bind("<Double-Button-1>",
                       lambda _e, p=display_path: self._open_fullview(p))

        # Key name (trimmed)
        name = label if len(label) <= 22 else label[:19] + "…"
        tk.Label(card, text=name, bg=BG_CARD, fg=C_DIM,
                 font=("Consolas", 7)).pack(pady=(2, 0))

        # File presence + hash distance
        ind_row = tk.Frame(card, bg=BG_CARD)
        ind_row.pack()
        own_col = C_GREEN if own else C_RED
        pub_col = C_GOLD  if pub else C_RED
        tk.Label(ind_row, text="Own ✓" if own else "Own ✗",
                 bg=BG_CARD, fg=own_col,
                 font=("Consolas", 7, "bold"), padx=3).pack(side=tk.LEFT)
        tk.Label(ind_row, text="Pub ✓" if pub else "Pub ✗",
                 bg=BG_CARD, fg=pub_col,
                 font=("Consolas", 7, "bold"), padx=3).pack(side=tk.LEFT)
        if paired and IMAGEHASH_AVAILABLE:
            dist_col = C_GREEN if dist <= 4 else C_TEAL if dist <= 10 else C_ORANGE
            tk.Label(ind_row, text=f"Δ{dist}",
                     bg=BG_CARD, fg=dist_col,
                     font=("Consolas", 7, "bold"), padx=3).pack(side=tk.LEFT)

        # Action buttons
        btn_frame = tk.Frame(card, bg=BG_CARD)
        btn_frame.pack(fill=tk.X, padx=4, pady=4)

        for btn_label, colour, folder in ACTIONS:
            tk.Button(
                btn_frame, text=btn_label,
                bg="#1c1c38", fg=colour,
                activebackground=colour, activeforeground=BG_DARK,
                relief=tk.FLAT, font=("Courier New", 8, "bold"),
                padx=4, pady=2, cursor="hand2",
                command=lambda l=btn_label, f=folder, o=own, p=pub, i=idx:
                    self._act(i, l, f, o, p)
            ).pack(side=tk.LEFT, padx=1)

        return card

    def _open_fullview(self, path: Optional[Path]):
        """Open a full-size overlay for the image. Double-click or Escape to close."""
        if not path or not path.exists():
            return

        # Load native image — cap at screen size but keep aspect ratio
        try:
            with Image.open(path) as raw:
                img = raw.convert("RGB")
                native_w, native_h = img.size
        except Exception:
            return

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        # Leave some margin so the window title bar stays accessible
        max_w = sw - 60
        max_h = sh - 80

        scale = min(1.0, max_w / native_w, max_h / native_h)
        disp_w = int(native_w * scale)
        disp_h = int(native_h * scale)

        with Image.open(path) as raw:
            img = raw.convert("RGB")
            if scale < 1.0:
                img = img.resize((disp_w, disp_h), Image.LANCZOS)

        # ── overlay window ────────────────────────────────────────────
        win = tk.Toplevel(self)
        win.title(path.name)
        win.configure(bg=BG_DARK)
        win.resizable(False, False)

        # Centre on screen
        x = (sw - disp_w) // 2
        y = (sh - disp_h) // 2
        win.geometry(f"{disp_w}x{disp_h + 26}+{x}+{y}")

        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo, bg=BG_DARK, cursor="hand2")
        lbl.image = photo          # keep reference
        lbl.pack()

        # Subtle info bar at the bottom
        info = (f"{path.name}   {native_w}×{native_h} px"
                + (f"  (shown at {int(scale*100)}%)" if scale < 1.0 else "  (native size)"))
        tk.Label(win, text=info, bg=BG_MID, fg=C_DIM,
                 font=("Consolas", 8), anchor=tk.W,
                 padx=8, pady=3).pack(fill=tk.X)

        def close(_e=None):
            win.destroy()

        lbl.bind("<Double-Button-1>", close)
        win.bind("<Escape>", close)
        win.bind("<Return>", close)
        win.focus_set()

    def _load_thumb(self, path: Optional[Path]) -> ImageTk.PhotoImage:
        try:
            if path and path.exists():
                with Image.open(path) as raw:
                    img = raw.convert("RGB")
                img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
                canvas = Image.new("RGB", THUMBNAIL_SIZE, (26, 26, 48))
                off = ((THUMBNAIL_SIZE[0] - img.width) // 2,
                       (THUMBNAIL_SIZE[1] - img.height) // 2)
                canvas.paste(img, off)
                return ImageTk.PhotoImage(canvas)
        except Exception:
            pass
        return ImageTk.PhotoImage(Image.new("RGB", THUMBNAIL_SIZE, (30, 30, 50)))

    # ── action handler ────────────────────────────────────────────────────────
    def _act(self, idx: int, label: str, folder: Optional[str],
             own: Optional[Path], pub: Optional[Path]):
        """
        Save    → move Own → Save/,  delete Pub
        Civit   → move Pub → Civit/, move Own → Save/
        Improve → move Own → Improve/, delete Pub
        Maybe   → move Own → Maybe/,  delete Pub
        No      → delete Own + Pub
        """
        if not self._folder:
            return

        errors = []

        def move_file(src: Optional[Path], dest_folder: str) -> bool:
            if src is None or not src.exists():
                return True  # nothing to do
            dest_dir = self._folder / dest_folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            # Avoid collision
            stem, sfx, n = dest.stem, dest.suffix, 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{n}{sfx}"
                n += 1
            try:
                shutil.move(str(src), str(dest))
                return True
            except Exception as e:
                errors.append(f"Move failed: {src.name} → {e}")
                return False

        def delete_file(src: Optional[Path]) -> bool:
            if src is None or not src.exists():
                return True
            try:
                src.unlink()
                return True
            except Exception as e:
                errors.append(f"Delete failed: {src.name} → {e}")
                return False

        if label == "Save":
            move_file(own, "Save")
            delete_file(pub)

        elif label == "Civit":
            move_file(pub, "Civit")
            move_file(own, "Save")

        elif label == "Improve":
            move_file(own, "Improve")
            delete_file(pub)

        elif label == "Maybe":
            move_file(own, "Maybe")
            delete_file(pub)

        elif label == "No":
            delete_file(own)
            delete_file(pub)

        if errors:
            messagebox.showerror("Error", "\n".join(errors))
            return

        # Remove card from grid
        self._remove_card(idx)
        self._done_count += 1
        remaining = len(self._pairs) - self._done_count
        self._counter_var.set(f"{remaining} remaining  ({self._done_count} done)")
        self._status_var.set(
            f"[{label}]  ✓  {(own or pub).name if (own or pub) else 'pair'}   "
            f"—  {remaining} remaining")

    def _remove_card(self, idx: int):
        """Visually collapse the card after action."""
        if idx < len(self._cards):
            card = self._cards[idx]
            # Grey it out and disable buttons
            self._dim_card(card)

    def _dim_card(self, card: tk.Frame):
        BG_DONE = "#0e0e1c"
        card.configure(highlightbackground="#2a2a44", bg=BG_DONE)
        for child in card.winfo_children():
            try:
                child.configure(bg=BG_DONE)
            except Exception:
                pass
            for sub in child.winfo_children():
                try:
                    sub.configure(bg=BG_DONE, state=tk.DISABLED,
                                  fg="#333355", disabledforeground="#333355")
                except Exception:
                    pass
        # Replace content with a done label
        for w in card.winfo_children():
            w.destroy()
        tk.Label(card, text="✓  done", bg=BG_DONE, fg="#333355",
                 font=("Courier New", 9), width=20, height=12).pack()


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()
