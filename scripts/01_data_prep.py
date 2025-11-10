# Simple check that Bilara JSONs can be read.
# Optional: save cleaned output to data/clean/

import sys
from pathlib import Path

# ensure 'lib' is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lib.bilara_io import iter_documents

BILARA_ROOT = "data"


def main():
    count = 0
    for doc in iter_documents(BILARA_ROOT):
        count += 1
        if count <= 5:  # only print first few
            print(f"{doc.doc_id}: {len(doc.segments)} segments")  # preview
    print(f"Total documents loaded: {count}")


if __name__ == "__main__":
    main()