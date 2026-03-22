#!/usr/bin/env python3
"""
ComfyUI / A1111 Metadata Extractor  v2
=======================================
Extracts CLIP Text Encoder prompts (positive / negative) from:
  - PNG/JPEG images with embedded ComfyUI API prompt JSON  (full link resolution)
  - PNG/JPEG images with embedded ComfyUI graph workflow JSON
  - PNG/JPEG images with A1111 / Forge "parameters" text chunk
  - Standalone .json workflow / API-prompt files
  - Plain .txt metadata files

Priority when multiple formats are present in the same image:
  1. ComfyUI API prompt  (PNG key "prompt")  – traces KSampler → CLIP chain
  2. ComfyUI graph workflow (PNG key "workflow")
  3. A1111/Forge parameters text (PNG key "parameters")
  4. EXIF UserComment (JPEG)

Output mirrors input folder/file naming exactly.
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARN] Pillow not installed. Image metadata extraction disabled.")
    print("       Install with: pip install Pillow\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Node class name sets
# ─────────────────────────────────────────────────────────────────────────────

CLIP_NODE_CLASSES = {
    "CLIPTextEncode",
    "CLIPTextEncodeSDXL",
    "CLIPTextEncodeSDXLRefiner",
    "BNK_CLIPTextEncodeAdvanced",
    "smZ CLIPTextEncode",
    "ImpactWildcardEncode",
    "WildcardEncode",
}

TEXT_SOURCE_CLASSES = {
    "Text Prompt (JPS)", "Textbox", "PrimitiveString",
    "ShowText|pysssss", "Note", "MarkdownNote",
}

TEXT_CONCAT_CLASSES = {
    "Text Concatenate (JPS)", "StringConcatenate", "JoinStrings",
    "Concat Strings", "TextConcatenate",
}

SAMPLER_CLASSES = {
    "KSampler", "KSamplerAdvanced", "ClownsharKSampler_Beta",
    "KSamplerSelect", "SamplerCustom",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Link / text resolution for ComfyUI API format
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_text(node_id: str, nodes: dict, depth: int = 0) -> str:
    """Follow ComfyUI API node links to resolve the final text string."""
    if depth > 15:
        return ""
    node = nodes.get(str(node_id))
    if not node:
        return ""

    ct = node.get("class_type", "")
    inputs = node.get("inputs", {})

    # Direct text holders
    if ct in CLIP_NODE_CLASSES | TEXT_SOURCE_CLASSES:
        val = inputs.get("text") or inputs.get("value") or ""
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, list) and val:
            return _resolve_text(str(val[0]), nodes, depth + 1)

    # Text concatenation nodes
    if ct in TEXT_CONCAT_CLASSES or "Concatenat" in ct:
        parts = []
        for key in sorted(inputs.keys()):
            if not key.startswith("text"):
                continue
            v = inputs[key]
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
            elif isinstance(v, list) and v:
                resolved = _resolve_text(str(v[0]), nodes, depth + 1)
                if resolved:
                    parts.append(resolved)
        return ", ".join(p for p in parts if p)

    # Fallback: plain "text" input string
    text_val = inputs.get("text")
    if isinstance(text_val, str) and text_val.strip():
        return text_val.strip()
    if isinstance(text_val, list) and text_val:
        return _resolve_text(str(text_val[0]), nodes, depth + 1)

    return ""


def _find_sampler_clip_ids(nodes: dict) -> tuple[set[str], set[str]]:
    """Trace KSampler positive/negative slots back to source node IDs."""
    pos_ids: set[str] = set()
    neg_ids: set[str] = set()
    for node_id, node in nodes.items():
        if node.get("class_type") not in SAMPLER_CLASSES:
            continue
        inputs = node.get("inputs", {})
        for slot, bucket in [("positive", pos_ids), ("negative", neg_ids)]:
            ref = inputs.get(slot)
            if isinstance(ref, list) and ref:
                bucket.add(str(ref[0]))
    return pos_ids, neg_ids


# ─────────────────────────────────────────────────────────────────────────────
#  ComfyUI API format  {"1": {"class_type": ..., "inputs": ...}, ...}
# ─────────────────────────────────────────────────────────────────────────────

def parse_comfyui_api(nodes: dict) -> dict[str, list[str]]:
    pos_ids, neg_ids = _find_sampler_clip_ids(nodes)
    positives: list[str] = []
    negatives: list[str] = []

    if pos_ids or neg_ids:
        for nid in sorted(pos_ids):
            text = _resolve_text(nid, nodes)
            if text:
                positives.append(text)
        for nid in sorted(neg_ids):
            text = _resolve_text(nid, nodes)
            if text:
                negatives.append(text)
    else:
        # No sampler found – classify by node title
        for node_id, node in nodes.items():
            if node.get("class_type") not in CLIP_NODE_CLASSES:
                continue
            title = (node.get("_meta", {}).get("title") or "").lower()
            text = _resolve_text(node_id, nodes)
            if not text:
                continue
            if "neg" in title or "negative" in title:
                negatives.append(text)
            else:
                positives.append(text)

    return {"positive": positives, "negative": negatives}


# ─────────────────────────────────────────────────────────────────────────────
#  ComfyUI graph/UI format  {"nodes": [...], "links": [...]}
# ─────────────────────────────────────────────────────────────────────────────

def parse_comfyui_graph(workflow: dict) -> dict[str, list[str]]:
    nodes: dict[str, dict] = {}
    for n in workflow.get("nodes", []):
        nid = str(n.get("id", ""))
        nodes[nid] = {
            "class_type": n.get("type", ""),
            "title":      n.get("title", ""),
            "inputs":     {},
            "_meta":      {"title": n.get("title", "")},
        }
        wv = n.get("widgets_values", [])
        if wv and isinstance(wv[0], str):
            nodes[nid]["inputs"]["text"] = wv[0]

    # Build link_id → (source_node_id, source_slot)
    link_map: dict[int, tuple[str, int]] = {}
    for link in workflow.get("links", []):
        if len(link) >= 4:
            link_map[link[0]] = (str(link[1]), link[2])

    # Wire up inputs for CLIP and sampler nodes
    for n in workflow.get("nodes", []):
        nid = str(n.get("id", ""))
        node_type = n.get("type", "")
        if node_type not in CLIP_NODE_CLASSES | SAMPLER_CLASSES:
            continue
        for inp in n.get("inputs", []):
            link_id = inp.get("link")
            inp_name = inp.get("name", "")
            if link_id and link_id in link_map:
                src_node, _ = link_map[link_id]
                nodes[nid]["inputs"][inp_name] = [src_node, 0]

    return parse_comfyui_api(nodes)


# ─────────────────────────────────────────────────────────────────────────────
#  A1111 / Forge plain-text
# ─────────────────────────────────────────────────────────────────────────────

def parse_a1111_text(text: str) -> dict[str, list[str]]:
    lines = text.strip().splitlines()
    positive_lines: list[str] = []
    negative_lines: list[str] = []
    mode = "positive"

    for line in lines:
        stripped = line.strip()
        if re.match(r"^Negative prompt\s*:", stripped, re.IGNORECASE):
            mode = "negative"
            after = re.sub(r"^Negative prompt\s*:\s*", "", stripped, flags=re.IGNORECASE)
            if after:
                negative_lines.append(after)
        elif re.match(
            r"^(Steps|Sampler|CFG|Seed|Size|Model|VAE|Clip skip|Denoising|Hires|Lora|Hash)\s*[\:,]",
            stripped, re.IGNORECASE,
        ):
            break
        elif mode == "positive" and stripped:
            positive_lines.append(stripped)
        elif mode == "negative" and stripped:
            negative_lines.append(stripped)

    pos = " ".join(positive_lines).strip()
    neg = " ".join(negative_lines).strip()
    return {
        "positive": [pos] if pos else [],
        "negative": [neg] if neg else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  File readers
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
JSON_EXTS  = {".json"}
TEXT_EXTS  = {".txt"}
ALL_EXTS   = IMAGE_EXTS | JSON_EXTS | TEXT_EXTS


def _is_api_format(data: dict) -> bool:
    return bool(data) and all(
        isinstance(v, dict) and "class_type" in v for v in data.values()
    )


def extract_from_image(path: Path) -> dict | None:
    if not PIL_AVAILABLE:
        return None
    try:
        img = Image.open(path)
        info = img.info

        # 1. ComfyUI API prompt JSON
        prompt_str = info.get("prompt")
        if prompt_str:
            try:
                data = json.loads(prompt_str)
                if _is_api_format(data):
                    result = parse_comfyui_api(data)
                    if result["positive"] or result["negative"]:
                        return {"source": "comfyui_api_prompt", "data": result, "raw": data}
            except (json.JSONDecodeError, AttributeError):
                pass

        # 2. ComfyUI graph workflow JSON
        workflow_str = info.get("workflow")
        if workflow_str:
            try:
                data = json.loads(workflow_str)
                if "nodes" in data:
                    result = parse_comfyui_graph(data)
                    src = "comfyui_graph_workflow"
                elif _is_api_format(data):
                    result = parse_comfyui_api(data)
                    src = "comfyui_api_workflow"
                else:
                    result = {"positive": [], "negative": []}
                    src = ""
                if result["positive"] or result["negative"]:
                    return {"source": src, "data": result, "raw": data}
            except (json.JSONDecodeError, AttributeError):
                pass

        # 3. A1111/Forge parameters text
        parameters = info.get("parameters") or info.get("comment")
        if parameters and isinstance(parameters, str):
            result = parse_a1111_text(parameters)
            if result["positive"] or result["negative"]:
                return {"source": "a1111_parameters", "data": result, "raw": parameters}

        # 4. EXIF UserComment
        try:
            exif_data = img._getexif() or {}
            user_comment = exif_data.get(0x9286)
            if user_comment:
                if isinstance(user_comment, bytes):
                    user_comment = user_comment.decode("utf-8", errors="ignore").lstrip("\x00")
                try:
                    data = json.loads(user_comment)
                    result = parse_comfyui_api(data) if _is_api_format(data) else parse_comfyui_graph(data)
                    if result["positive"] or result["negative"]:
                        return {"source": "exif_comfyui", "data": result, "raw": data}
                except json.JSONDecodeError:
                    result = parse_a1111_text(user_comment)
                    if result["positive"] or result["negative"]:
                        return {"source": "exif_a1111", "data": result, "raw": user_comment}
        except Exception:
            pass

    except Exception as e:
        print(f"  [WARN] Could not read image {path.name}: {e}")
    return None


def extract_from_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "nodes" in data:
            result = parse_comfyui_graph(data)
            src = "json_graph_workflow"
        elif _is_api_format(data):
            result = parse_comfyui_api(data)
            src = "json_api_prompt"
        else:
            return None
        if result["positive"] or result["negative"]:
            return {"source": src, "data": result, "raw": data}
    except Exception as e:
        print(f"  [WARN] Could not parse JSON {path.name}: {e}")
    return None


def extract_from_txt(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            data = json.loads(text)
            result = parse_comfyui_api(data) if _is_api_format(data) else parse_comfyui_graph(data)
            if result["positive"] or result["negative"]:
                return {"source": "txt_json", "data": result, "raw": data}
        except json.JSONDecodeError:
            pass
        result = parse_a1111_text(text)
        if result["positive"] or result["negative"]:
            return {"source": "txt_a1111", "data": result, "raw": text}
    except Exception as e:
        print(f"  [WARN] Could not read text {path.name}: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_output_path(
    input_path: Path, input_root: Path, output_root: Path, suffix: str
) -> Path:
    rel = input_path.relative_to(input_root)
    out = output_root / rel.parent / (rel.stem + suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_txt(path: Path, data: dict[str, list[str]]) -> None:
    pos = "\n\n---\n\n".join(data["positive"]) if data["positive"] else "(none)"
    neg = "\n\n---\n\n".join(data["negative"]) if data["negative"] else "(none)"
    lines = [
        "=== POSITIVE PROMPT ===",
        pos,
        "",
        "=== NEGATIVE PROMPT ===",
        neg,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, meta: dict, source_path: Path) -> None:
    out = {
        "source_file": str(source_path),
        "extraction_source": meta["source"],
        "positive": meta["data"]["positive"],
        "negative": meta["data"]["negative"],
    }
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  Processing
# ─────────────────────────────────────────────────────────────────────────────

def _extract(path: Path) -> dict | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return extract_from_image(path)
    if ext in JSON_EXTS:
        return extract_from_json(path)
    if ext in TEXT_EXTS:
        return extract_from_txt(path)
    return None


def process_directory(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = True,
    skip_empty: bool = True,
) -> None:
    pattern = "**/*" if recursive else "*"
    files = sorted(
        f for f in input_dir.glob(pattern)
        if f.is_file() and f.suffix.lower() in ALL_EXTS
    )
    if not files:
        print(f"No supported files found in {input_dir}")
        return

    ok = skip = 0
    for file in files:
        print(f"  {file.relative_to(input_dir)}", end="  ")
        meta = _extract(file)
        if meta is None or (
            skip_empty and not meta["data"]["positive"] and not meta["data"]["negative"]
        ):
            print("→ SKIP")
            skip += 1
            continue
        write_txt(build_output_path(file, input_dir, output_dir, ".txt"), meta["data"])
        write_json(build_output_path(file, input_dir, output_dir, ".json"), meta, file)
        print(f"→ OK [{meta['source']}]")
        ok += 1

    print(f"\n{'─'*52}")
    print(f"Done.  OK: {ok}  |  Skipped: {skip}")
    print(f"Output: {output_dir.resolve()}")


def process_single_file(path: Path, output_dir: Path) -> None:
    meta = _extract(path)
    if meta is None:
        print(f"No prompts found in {path.name}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    txt_out  = output_dir / (path.stem + ".txt")
    json_out = output_dir / (path.stem + ".json")
    write_txt(txt_out, meta["data"])
    write_json(json_out, meta, path)

    print(f"Extracted: {path.name}  [{meta['source']}]")
    print(f"  TXT  → {txt_out}")
    print(f"  JSON → {json_out}")
    print()
    print("=== POSITIVE ===")
    for p in meta["data"]["positive"]:
        print(p[:300], "..." if len(p) > 300 else "")
        print()
    print("=== NEGATIVE ===")
    for n in meta["data"]["negative"]:
        print(n[:300], "..." if len(n) > 300 else "")
        print()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ComfyUI / A1111 CLIP prompts from images & JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  .png .jpg .jpeg .webp  – ComfyUI API prompt, graph workflow, or A1111 parameters
  .json                  – ComfyUI API prompt or graph workflow
  .txt                   – A1111/Forge parameters or JSON

Examples:
  python comfy_meta_extractor.py ./renders ./prompts_out
  python comfy_meta_extractor.py ./image.png ./out
  python comfy_meta_extractor.py ./renders ./out --no-recursive
  python comfy_meta_extractor.py ./renders ./out --include-empty
""",
    )
    parser.add_argument("input",  help="Input file or directory")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--no-recursive",  dest="recursive",  action="store_false")
    parser.add_argument("--include-empty", dest="skip_empty", action="store_false")
    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}")
        sys.exit(1)

    if input_path.is_dir():
        process_directory(input_path, output_path,
                          recursive=args.recursive, skip_empty=args.skip_empty)
    else:
        process_single_file(input_path, output_path)


if __name__ == "__main__":
    main()
