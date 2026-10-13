"""Command-line interface for LinguaLoom."""
from __future__ import annotations

import re
import sys
import sqlite3
import hashlib
import os
import json
import time
from pathlib import Path

import requests
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString, Tag
from copy import copy
from tqdm import tqdm

AUTHOR = "Bruno F Blini"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def choose_epub() -> Path:
    """Choose an EPUB with a file picker or relative/absolute path input."""
    print(f"Current working directory: {Path.cwd()}")
    print("Enter a filename (the .epub extension is optional), relative/absolute path, or BROWSE.")
    while True:
        raw = ask("Original EPUB path", "BROWSE").strip()
        if raw.upper() == "BROWSE":
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                selected = filedialog.askopenfilename(
                    title="Choose original EPUB",
                    filetypes=[("EPUB books", "*.epub"), ("All files", "*.*")],
                )
                root.destroy()
                if selected:
                    return Path(selected).expanduser().resolve()
                print("No file selected. Enter a path or choose BROWSE again.")
                continue
            except Exception:
                print("File picker unavailable; enter a path manually.")
                continue
        raw = raw.strip('"').strip("'")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        # Convenience: accept "book" when the actual file is "book.epub".
        candidates = [path]
        if not path.suffix:
            candidates.append(Path(f"{path}.epub"))
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
        if not path.exists():
            try:
                pattern = raw if Path(raw).suffix else f"{raw}.epub"
                matches = sorted(Path.cwd().glob(pattern))
            except (ValueError, OSError):
                matches = []
            if matches:
                print("Matching paths:")
                for candidate in matches[:15]:
                    print(f"  {candidate}")
            else:
                epubs = sorted(Path.cwd().glob("*.epub"))
                if epubs:
                    print("EPUB files in the current directory:")
                    for candidate in epubs[:20]:
                        print(f"  {candidate.name}")
                else:
                    print("Path not found. Relative paths are resolved from the current working directory.")
            continue
        if not path.is_file() or path.suffix.lower() != ".epub":
            print("Please choose an .epub file (the extension may be omitted when entering its path).")
            continue
        return path.resolve()


def ollama_models() -> list[dict]:
    """Return Ollama model metadata so we can automatically choose the newest Qwen family."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return response.json().get("models", [])
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. "
            "Make sure Ollama is installed and running."
        ) from exc


def latest_qwen_model(models: list[dict]) -> str | None:
    """Select the newest installed Qwen generation, breaking ties by latest pull time."""
    candidates = []
    for item in models:
        name = item.get("name", "")
        if "qwen" not in name.lower():
            continue
        match = re.search(r"qwen[-_ ]?(\d+(?:\.\d+)*)", name, re.IGNORECASE)
        version = tuple(int(part) for part in match.group(1).split(".")) if match else (0,)
        # Ollama reports RFC3339 timestamps in modified_at for locally available models.
        modified = str(item.get("modified_at", ""))
        candidates.append((version, modified, name))
    if not candidates:
        return None
    return max(candidates)[2]


def replace_checkpoint(checkpoint: Path, output: Path, attempts: int = 8) -> None:
    """Replace the output atomically, retrying transient Windows file-lock errors."""
    last_error = None
    for attempt in range(attempts):
        try:
            os.replace(checkpoint, output)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(0.25 * (attempt + 1), 1.5))
    raise PermissionError(
        f"Windows could not replace the output EPUB because a file is locked. "
        f"Close the EPUB in any reader/preview pane and retry. The complete checkpoint "
        f"has been preserved at: {checkpoint}"
    ) from last_error


def ollama_generate(model: str, prompt: str, timeout: int = 180) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "keep_alive": "10m", "think": False, "options": {"temperature": 0.1, "num_predict": 2048}},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def language_check(model: str, language: str, sample: str) -> tuple[bool, str]:
    prompt = f"""Identify the language of the text sample. Do not translate it.
Reply in exactly this format:
LANGUAGE: <language name>
MATCH: YES or NO

The user's claimed language is: {language}
Use MATCH: YES only if the sample is predominantly written in that language.
If it is too short or ambiguous, use MATCH: NO.

TEXT SAMPLE:
{sample[:3500]}
"""
    answer = ollama_generate(model, prompt)
    match = re.search(r"LANGUAGE:\s*(.+)", answer, re.IGNORECASE)
    detected = match.group(1).strip() if match else "uncertain"
    yes = re.search(r"MATCH:\s*YES\b", answer, re.IGNORECASE) is not None
    return yes, detected


def translate_paragraph(model: str, source_language: str, target_language: str, text: str) -> str:
    prompt = f"""Translate the passage from {source_language} into {target_language} for language learners.

Rules:
- Stay as close to the source wording, meaning, sentence structure, and order as natural comprehension allows.
- Prefer a literal, transparent translation over elegant or literary phrasing.
- Preserve paragraph meaning, names, numbers, punctuation, and sentence boundaries.
- Do not summarize, explain, add notes, add alternatives, or include the original text.
- Return only the translation.
- If the passage is already in {target_language} or contains no translatable prose, reproduce it faithfully.

PASSAGE:
{text}
"""
    result = ollama_generate(model, prompt)
    return result or "[Translation unavailable]"


def translate_batch(model: str, source_language: str, target_language: str, paragraphs: list[str]) -> list[str]:
    """Translate several paragraphs in one call while preserving paragraph boundaries."""
    payload = [{"id": i, "text": value} for i, value in enumerate(paragraphs, start=1)]
    prompt = f"""Translate each item from {source_language} into {target_language} for language learners.
Return ONLY a valid JSON array of objects with exactly these keys: id, translation.
Include one object for every input id, in the same order. Do not merge, omit, explain, or add notes.
Keep each translation close to the original wording and sentence structure.

INPUT JSON:
{json.dumps(payload, ensure_ascii=False)}
"""
    raw = ollama_generate(model, prompt, timeout=300)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, dict):
            decoded = decoded.get("translations", decoded.get("items"))
        if not isinstance(decoded, list):
            raise ValueError("Batch response was not a JSON list")
        found = {}
        for item in decoded:
            if isinstance(item, dict) and "id" in item and "translation" in item:
                found[int(item["id"])] = str(item["translation"]).strip()
        if len(found) != len(paragraphs) or any(not found.get(i) for i in range(1, len(paragraphs) + 1)):
            raise ValueError("Batch response omitted one or more translations")
        return [found[i] for i in range(1, len(paragraphs) + 1)]
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Qwen returned an incomplete batch; retrying paragraphs individually") from exc


class TranslationCache:
    """Persistent SQLite cache; successful paragraph translations survive crashes."""
    def __init__(self, path: Path, model: str, source_language: str, target_language: str):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS translations (
            cache_key TEXT PRIMARY KEY,
            translation TEXT NOT NULL,
            model TEXT NOT NULL,
            source_language TEXT NOT NULL,
            target_language TEXT NOT NULL,
            source_text TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        self.connection.commit()
        self.model = model
        self.source_language = source_language
        self.target_language = target_language

    def _key(self, text: str) -> str:
        raw = "\0".join((self.model, self.source_language.casefold(), self.target_language.casefold(), text))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str) -> str | None:
        row = self.connection.execute(
            "SELECT translation FROM translations WHERE cache_key = ?", (self._key(text),)
        ).fetchone()
        return row[0] if row else None

    def put(self, text: str, translation: str) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO translations
            (cache_key, translation, model, source_language, target_language, source_text, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (self._key(text), translation, self.model, self.source_language,
             self.target_language, text),
        )
        # Commit every completed paragraph so a crash loses at most the current request.
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def is_translatable_text(tag: Tag) -> bool:
    return tag.name in {"p", "blockquote", "li", "h1", "h2", "h3", "h4", "h5", "h6", "figcaption", "dt", "dd"}


def extract_sample(book: epub.EpubBook, limit: int = 12) -> str:
    chunks: list[str] = []
    for item in book.get_items():
        if item.get_type() != 9:  # ebooklib.ITEM_DOCUMENT
            continue
        content = item.get_content()
        if not content:
            continue
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup.find_all(is_translatable_text):
            text = tag.get_text(" ", strip=True)
            if len(text) > 25:
                chunks.append(text)
            if len(chunks) >= limit:
                return "\n".join(chunks)
    return "\n".join(chunks)


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "language"


def collect_translatable_tags(body: Tag) -> list[Tag]:
    """Return translatable blocks, excluding blocks nested inside another block."""
    tags = list(body.find_all(is_translatable_text))
    ids = {id(tag) for tag in tags}
    result = []
    for tag in tags:
        if any(isinstance(parent, Tag) and parent is not body and id(parent) in ids for parent in tag.parents):
            continue
        if tag.get_text(" ", strip=True):
            result.append(tag)
    return result


def count_translatable_blocks(book: epub.EpubBook) -> int:
    total = 0
    for item in book.get_items():
        if item.get_type() != 9:
            continue
        content = item.get_content()
        if not content:
            continue
        soup = BeautifulSoup(content, "html.parser")
        if soup.body:
            total += len(collect_translatable_tags(soup.body))
    return total


def sanitize_epub_items_for_write(book: epub.EpubBook) -> list[str]:
    """Replace missing payloads on EPUB resources with empty bytes before serialization.

    Some EPUBs contain manifest entries whose payload is missing. EbookLib can pass
    those None values to its ZIP writer, which raises a low-level TypeError. Report
    the affected item names so the source can be inspected if the book still fails.
    """
    repaired: list[str] = []
    for item in book.get_items():
        try:
            payload = item.get_content()
        except Exception as exc:
            raise RuntimeError(
                f"Could not read EPUB resource {item.get_name()!r} "
                f"({type(item).__name__}): {exc}"
            ) from exc
        if payload is None:
            setter = getattr(item, "set_content", None)
            if not callable(setter):
                raise RuntimeError(
                    f"EPUB resource {item.get_name()!r} has no content and cannot be repaired."
                )
            setter(b"")
            repaired.append(item.get_name())
    return repaired


def build_bilingual_html(item, model, source_language, target_language, cache, progress,
                        batch_size: int = 5, max_batch_chars: int = 4500) -> bytes:
    # EbookLib can expose empty/malformed document items with None content. Passing
    # that value back to set_content() raises: "Argument must be bytes or unicode,
    # got 'NoneType'". Keep every returned document payload a real byte string.
    content = item.get_content()
    if content is None or content == b"" or content == "":
        return b'<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title></title></head><body></body></html>'
    soup = BeautifulSoup(content, "html.parser")
    body = soup.body
    if body is None:
        # Preserve documents without a <body> when possible, while normalizing
        # their content to bytes for EbookLib's set_content().
        serialized = str(soup).encode("utf-8")
        return serialized or b'<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head></head><body></body></html>'

    style = soup.new_tag("style", type="text/css")
    style.string = """
    /* A fixed 50/50 table layout is more reliable in EPUB readers than CSS Grid. */
    .ll-row { display: table; width: 100%; table-layout: fixed; border-collapse: separate; border-spacing: 0; margin: 0 0 1em; page-break-inside: avoid; break-inside: avoid; }
    .ll-original, .ll-translation { display: table-cell; width: 50%; min-width: 0; vertical-align: top; box-sizing: border-box; overflow-wrap: anywhere; padding: 0 .75em 0 0; }
    .ll-translation { border-left: 1px solid #aaa; padding: 0 0 0 .75em; }
    .ll-label { font-size: .75em; font-weight: bold; text-transform: uppercase; letter-spacing: .04em; color: #666; margin-bottom: .35em; }
    .ll-content { margin: 0; max-width: 100%; }
    .ll-original img, .ll-translation img, .ll-media-row img { max-width: 100%; height: auto; }
    .ll-media-row { page-break-inside: avoid; break-inside: avoid; }
    """
    head = soup.head
    if head is None and soup.find("html"):
        head = soup.new_tag("head")
        soup.find("html").insert(0, head)
    if head is not None:
        head.append(style)

    tags = collect_translatable_tags(body)
    texts = [tag.get_text(" ", strip=True) for tag in tags]
    missing_indices = [i for i, value in enumerate(texts) if value and cache.get(value) is None]
    cached_count = sum(1 for value in texts if value and cache.get(value) is not None)
    progress.update(cached_count)

    batches, current, current_chars = [], [], 0
    for idx in missing_indices:
        size = len(texts[idx])
        if current and (len(current) >= batch_size or current_chars + size > max_batch_chars):
            batches.append(current)
            current, current_chars = [], 0
        current.append(idx)
        current_chars += size
    if current:
        batches.append(current)

    for batch in batches:
        progress.set_postfix_str(f"Translating batch ({len(batch)} paragraphs)")
        batch_texts = [texts[i] for i in batch]
        try:
            translations = translate_batch(model, source_language, target_language, batch_texts)
        except (ValueError, requests.RequestException, RuntimeError) as exc:
            print(f"\nBatch fallback to individual paragraphs: {exc}", flush=True)
            translations = [translate_paragraph(model, source_language, target_language, source)
                            for source in batch_texts]
        for idx, translated in zip(batch, translations):
            cache.put(texts[idx], translated)
            progress.update(1)

    # Images not contained in a translatable text block (e.g. standalone figures)
    # get their own paired row so the illustration is visible in both book columns.
    translatable_ids = {id(tag) for tag in tags}
    standalone_images = []
    for image_tag in list(body.find_all(["img"])):
        if any(isinstance(parent, Tag) and id(parent) in translatable_ids for parent in image_tag.parents):
            continue
        standalone_images.append(image_tag)
    for image_tag in standalone_images:
        row = soup.new_tag("div", attrs={"class": "ll-row ll-media-row"})
        left = soup.new_tag("div", attrs={"class": "ll-original"})
        right = soup.new_tag("div", attrs={"class": "ll-translation"})
        left.append(copy(image_tag))
        right.append(copy(image_tag))
        row.append(left)
        row.append(right)
        image_tag.replace_with(row)

    for tag, source_text in zip(tags, texts):
        if not source_text:
            continue
        translated = cache.get(source_text)
        if translated is None:
            # Defensive fallback; persist immediately to support resume after failure.
            translated = translate_paragraph(model, source_language, target_language, source_text)
            cache.put(source_text, translated)
            progress.update(1)

        row = soup.new_tag("div", attrs={"class": "ll-row"})
        left = soup.new_tag("div", attrs={"class": "ll-original"})
        left_label = soup.new_tag("div", attrs={"class": "ll-label"})
        left_label.string = source_language
        left.append(left_label)
        original_clone = soup.new_tag(tag.name, attrs={"class": "ll-content"})
        for child in tag.contents:
            original_clone.append(copy(child))
        left.append(original_clone)

        right = soup.new_tag("div", attrs={"class": "ll-translation"})
        right_label = soup.new_tag("div", attrs={"class": "ll-label"})
        right_label.string = target_language
        right.append(right_label)
        translation_tag = soup.new_tag("p", attrs={"class": "ll-content"})
        translation_tag.string = translated
        # If the source paragraph contains inline illustrations, repeat them on the
        # translation side as well as keeping them in the original-side clone.
        for inline_image in original_clone.find_all(["img"]):
            right.append(copy(inline_image))
        right.append(translation_tag)
        row.append(left)
        row.append(right)
        tag.replace_with(row)

    return str(soup).encode("utf-8")


def create_translated_epub(
    source_path: Path,
    target_language: str,
    source_language: str,
    model: str,
) -> Path:
    book = epub.read_epub(str(source_path))
    output = source_path.with_name(f"{source_path.stem}_translated.epub")
    # Unique same-directory temporary file: avoids stale checkpoint collisions and
    # permits an atomic rename on Windows once EbookLib has closed the archive.
    checkpoint = output.with_name(f".{output.stem}.{os.getpid()}.partial.epub")
    cache_path = source_path.with_name(f"{source_path.stem}_lingualoom_cache.sqlite3")

    docs = [item for item in book.get_items() if item.get_type() == 9]
    if not docs:
        raise RuntimeError("No readable XHTML/HTML content was found in this EPUB.")
    total_blocks = count_translatable_blocks(book)
    if total_blocks == 0:
        raise RuntimeError("No readable paragraphs or text blocks were found in this EPUB.")

    print(f"Found {total_blocks:,} text blocks across {len(docs):,} chapter/content files.", flush=True)
    print(f"Persistent translation cache: {cache_path}", flush=True)
    print("Each chapter is processed and checkpointed before moving to the next.", flush=True)
    cache = TranslationCache(cache_path, model, source_language, target_language)
    try:
        with tqdm(
            total=total_blocks,
            desc="Translating",
            unit="paragraphs",
            ncols=100,
            dynamic_ncols=False,
            bar_format="{desc} |{bar:28}| {percentage:3.0f}% ({n_fmt}/{total_fmt} paragraphs) [{elapsed}<{remaining}]",
        ) as progress:
            for index, item in enumerate(docs, start=1):
                progress.set_description_str(f"Chapter {index}/{len(docs)}")
                # Build exactly one chapter at a time. The SQLite cache is committed
                # after each successful translation, then the partially completed EPUB
                # is atomically replaced with a fresh checkpoint after this chapter.
                translated_content = build_bilingual_html(
                    item, model, source_language, target_language, cache, progress
                )
                if translated_content is None:
                    raise RuntimeError(
                        f"Internal error: content builder returned None for {item.get_name()}"
                    )
                item.set_content(translated_content)
                identifiers = book.get_metadata("DC", "identifier")
                original_identifier = identifiers[0][0] if identifiers else source_path.stem
                book.set_identifier(f"{original_identifier}-lingualoom")
                book.add_metadata("DC", "creator", AUTHOR)
                book.add_metadata("DC", "description", f"Bilingual edition: {source_language} / {target_language}. Created with LinguaLoom.")
                book.add_metadata("DC", "language", safe_slug(source_language))
                repaired_items = sanitize_epub_items_for_write(book)
                if repaired_items:
                    print(
                        "\nWarning: repaired missing payloads for EPUB resources: "
                        + ", ".join(repaired_items),
                        flush=True,
                    )
                try:
                    epub.write_epub(str(checkpoint), book)
                except Exception as exc:
                    # Include resource diagnostics instead of only the generic
                    # bytes/unicode/NoneType message from the ZIP/XML writer.
                    none_items = []
                    for resource in book.get_items():
                        try:
                            if resource.get_content() is None:
                                none_items.append(resource.get_name())
                        except Exception:
                            none_items.append(resource.get_name() + " (content read failed)")
                    detail = (
                        f"; resources still reporting missing content: {', '.join(none_items)}"
                        if none_items else "; no resources report missing content"
                    )
                    raise RuntimeError(
                        f"EbookLib failed while writing the EPUB checkpoint{detail}. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ) from exc
                replace_checkpoint(checkpoint, output)
                print(f"Checkpoint saved after chapter {index}/{len(docs)}.", flush=True)
    finally:
        cache.close()
        # Do not delete a checkpoint after a failed Windows rename: it may be the
        # only complete recovery copy of the translated chapters.

    print(f"Processed {len(docs)} content files. Translation cache saved at: {cache_path}")
    print("You can keep the cache file for future retries; it avoids repeating completed translations.")
    return output


def main() -> None:
    print("\nLinguaLoom — side-by-side language-learning EPUBs")
    print("Author: Bruno F Blini\n")
    try:
        available_models = ollama_models()
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    model = latest_qwen_model(available_models)
    if not model:
        print("No Qwen model found in Ollama.")
        print("Install one first, for example: ollama pull qwen3:8b")
        sys.exit(1)
    print(f"Automatically selected latest installed Qwen model: {model}")

    source_path = choose_epub()
    source_language = ask("What is the original language (e.g. Portuguese, French, Japanese)")
    if not source_language:
        print("The original language is required.")
        sys.exit(1)
    target_language = ask("Translate into which language", "English")

    print("\nChecking the EPUB language against your answer...")
    try:
        book = epub.read_epub(str(source_path))
        sample = extract_sample(book)
        if not sample.strip():
            print("Could not find enough readable text to check the language.")
            sys.exit(1)
        matches, detected = language_check(model, source_language, sample)
        print(f"Qwen identifies the sample as: {detected}")
        if not matches:
            print("The sample may not match the language you entered.")
            confirm = ask("Continue anyway? Type 'yes' to proceed", "no").lower()
            if confirm not in {"yes", "y"}:
                print("Cancelled. Check the source language or EPUB and try again.")
                sys.exit(0)
        else:
            print("Language check passed.")
        output = create_translated_epub(source_path, target_language, source_language, model)
        print(f"\nDone! Your bilingual EPUB is here:\n{output}")
    except requests.RequestException as exc:
        print(f"\nOllama request failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nCould not create the translated EPUB ({type(exc).__name__}): {exc}")
        print("If this happens again, please share the full error above and the chapter shown in the progress bar.")
        sys.exit(1)


if __name__ == "__main__":
    main()
