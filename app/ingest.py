import hashlib, re
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf", ".docx"}

def read_document(path: Path) -> str:
    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if path.suffix.lower() == ".docx":
        from docx import Document
        return "\\n".join(p.text for p in Document(str(path)).paragraphs)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def clean_text(text: str) -> str:
    return re.sub(r"\\n{3,}", "\\n\\n", re.sub(r"[ \\t]+", " ", text.replace("\\x00", " "))).strip()

def chunk_text(text: str, chunk_size=900, overlap=150) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    text = clean_text(text)
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + int(chunk_size * .65), end)
            if boundary > start: end = boundary
        piece = text[start:end].strip()
        if piece: chunks.append(piece)
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return chunks

def load_chunks(root, chunk_size=900, overlap=150):
    root = Path(root)
    if not root.exists(): raise FileNotFoundError(f"Knowledge directory missing: {root}")
    records = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED):
        for i, chunk in enumerate(chunk_text(read_document(path), chunk_size, overlap)):
            rel = str(path.relative_to(root))
            digest = hashlib.sha1(f"{rel}:{i}:{chunk}".encode()).hexdigest()[:12]
            records.append({"chunk_id": f"{path.stem}-{i:04d}-{digest}", "source": rel, "text": chunk})
    return records
