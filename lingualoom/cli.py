"""Command-line interface for LinguaLoom."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import requests
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString, Tag

AUTHOR = "Bruno F Blini"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def choose_epub() -> Path:
    while True:
        raw = ask("Path to the original EPUB file")
        path = Path(raw.strip('"').strip("'")).expanduser()
        if not path.exists():
            print("That path does not exist. Try again.")
            continue
        if not path.is_file() or path.suffix.lower() != ".epub":
            print("Please choose an .epub file.")
            continue
        return path.resolve()


def ollama_tags() -> list[str]:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return [item.get("name", "") for item in response.json().get("models", [])]
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. "
            "Make sure Ollama is installed and running."
        ) from exc


def ollama_generate(model: str, prompt: str, timeout: int = 180) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
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


def is_translatable_text(tag: Tag) -> bool:
    return tag.name in {"p", "blockquote", "li", "h1", "h2", "h3", "h4", "h5", "h6", "figcaption", "dt", "dd"}


def extract_sample(book: epub.EpubBook, limit: int = 12) -> str:
    chunks: list[str] = []
    for item in book.get_items():
        if item.get_type() != 9:  # ebooklib.ITEM_DOCUMENT
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup.find_all(is_translatable_text):
            text = tag.get_text(" ", strip=True)
            if len(text) > 25:
                chunks.append(text)
            if len(chunks) >= limit:
                return "\n".join(chunks)
    return "\n".join(chunks)


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "language"


def build_bilingual_html(
    item: epub.EpubHtml,
    model: str,
    source_language: str,
    target_language: str,
    cache: dict[str, str],
) -> bytes:
    soup = BeautifulSoup(item.get_content(), "html.parser")
    body = soup.body
    if body is None:
        return item.get_content()

    # Add styles locally to each XHTML document so readers do not need external assets.
    style = soup.new_tag("style", type="text/css")
    style.string = """
    .ll-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
              gap: 1.1em; align-items: start; margin: 0 0 1em; }
    .ll-original, .ll-translation { min-width: 0; overflow-wrap: anywhere; }
    .ll-translation { border-left: 1px solid #aaa; padding-left: 1em; }
    .ll-label { font-size: .75em; font-weight: bold; text-transform: uppercase;
                letter-spacing: .04em; color: #666; margin-bottom: .35em; }
    .ll-content { margin: 0; }
    @media (max-width: 560px) {
      .ll-row { grid-template-columns: 1fr; gap: .35em; }
      .ll-translation { border-left: 0; border-top: 1px solid #aaa; padding: .5em 0 0; }
    }
    """
    head = soup.head
    if head is None:
        head = soup.new_tag("head")
        html_tag = soup.find("html")
        if html_tag:
            html_tag.insert(0, head)
    if head is not None:
        head.append(style)

    # Replace text-bearing blocks with a two-column row, keeping their inner markup.
    candidates = list(body.find_all(is_translatable_text))
    for tag in candidates:
        if tag.find_parent(class_="ll-row") is not None:
            continue
        source_text = tag.get_text(" ", strip=True)
        if not source_text:
            continue
        # Avoid translating nested blocks twice (e.g. paragraphs inside blockquotes).
        if any(parent in candidates for parent in tag.parents if isinstance(parent, Tag) and parent is not body):
            continue
        if source_text not in cache:
            cache[source_text] = translate_paragraph(model, source_language, target_language, source_text)
        translated = cache[source_text]

        row = soup.new_tag("div", attrs={"class": "ll-row"})
        left = soup.new_tag("div", attrs={"class": "ll-original"})
        left_label = soup.new_tag("div", attrs={"class": "ll-label"})
        left_label.string = source_language
        left.append(left_label)
        original_clone = soup.new_tag(tag.name, attrs={"class": "ll-content"})
        # Keep the source content's markup, not just its flattened text.
        for child in list(tag.contents):
            original_clone.append(child.extract() if isinstance(child, NavigableString) else child.extract())
        left.append(original_clone)

        right = soup.new_tag("div", attrs={"class": "ll-translation"})
        right_label = soup.new_tag("div", attrs={"class": "ll-label"})
        right_label.string = target_language
        right.append(right_label)
        translation_tag = soup.new_tag("p", attrs={"class": "ll-content"})
        translation_tag.string = translated
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
    cache: dict[str, str] = {}
    output = source_path.with_name(f"{source_path.stem}_translated.epub")

    docs = [item for item in book.get_items() if item.get_type() == 9]
    if not docs:
        raise RuntimeError("No readable XHTML/HTML content was found in this EPUB.")

    total = 0
    for item in docs:
        print(f"Processing chapter: {item.get_name()}")
        item.set_content(build_bilingual_html(item, model, source_language, target_language, cache))
        total += 1

    book.set_identifier(f"{book.get_metadata('DC', 'identifier')[0][0] if book.get_metadata('DC', 'identifier') else source_path.stem}-lingualoom")
    book.add_metadata("DC", "creator", AUTHOR)
    book.add_metadata("DC", "description", f"Bilingual edition: {source_language} / {target_language}. Created with LinguaLoom.")
    book.add_metadata("DC", "language", safe_slug(source_language))
    epub.write_epub(str(output), book)
    print(f"Processed {total} content documents; translated {len(cache)} unique text blocks.")
    return output


def main() -> None:
    print("\nLinguaLoom — side-by-side language-learning EPUBs")
    print("Author: Bruno F Blini\n")
    try:
        available = ollama_tags()
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    qwen_models = [name for name in available if "qwen" in name.lower()]
    if not qwen_models:
        print("No Qwen model found in Ollama.")
        print("Install one first, for example: ollama pull qwen2.5:7b")
        sys.exit(1)
    default_model = next((name for name in qwen_models if name.startswith("qwen2.5:7b")), qwen_models[0])
    print("Available Qwen models:")
    for name in qwen_models:
        print(f"  - {name}")
    model = ask("Which Qwen model should LinguaLoom use", default_model)
    if model not in available:
        print(f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}")
        sys.exit(1)

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
        print(f"\nCould not create the translated EPUB: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
