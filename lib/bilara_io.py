import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional


# Represents one logical sutta (a Bilara translation).
@dataclass
class Document:
    doc_id: str
    title: str
    segments: Dict[str, str]   # key: "mn1:1.1" -> text
    sc_link: str


def build_sc_link(doc_id: str, lang: str = "en", translator: str = "sujato") -> str:
    # Build SuttaCentral link for a doc_id.
    return f"https://suttacentral.net/{doc_id}/{lang}/{translator}"


def find_files(bilara_root: str, patterns: Optional[List[str]] = None) -> List[Path]:
    # Collect Bilara translation JSON file paths. If patterns is None, use sensible defaults.
    root = Path(bilara_root)
    if patterns is None:
        patterns = [
            "translation/en/sujato/sutta/dn",
            "translation/en/sujato/sutta/mn",
            # more as needed
        ]

    files: List[Path] = []
    for base in patterns:
        base_path = root / base
        if not base_path.exists():
            continue
        files.extend(base_path.rglob("*.json"))
    return files


def load_documents(path: Path) -> Document:
    # Load one Bilara JSON -> Document. Assumes keys like "mn1:1.1".
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError(f"Empty bilara file: {path}")

    # doc_id = prefix before the first ':' in any key
    first_key = next(iter(data.keys()))
    doc_id = first_key.split(":")[0]
    title = doc_id.upper()  # simple title

    sc_link = build_sc_link(doc_id)
    return Document(doc_id=doc_id, title=title, segments=data, sc_link=sc_link)


def iter_documents(bilara_root: str) -> Iterator[Document]:
    # Generator: iterate over found translation JSONs and yield Document objects.
    for path in find_files(bilara_root):
        try:
            yield load_documents(path)
        except Exception as e:
            # warn and continue on error
            print(f"[WARN] Could not load {path}: {e}")
