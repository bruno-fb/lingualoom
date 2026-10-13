# LinguaLoom

**LinguaLoom** creates a bilingual EPUB for language learners: the original text appears on the left, and a close, learner-friendly translation appears on the right. It uses a locally running [Ollama](https://ollama.com/) model from the Qwen family, so the text is processed by your local model rather than a hosted translation API.

**Author:** Bruno F Blini

## Version 1.1.1

- Handles empty or malformed EPUB content items without passing `None` to EbookLib.
- Uses a fixed-width, cleaner progress bar showing chapter, percentage, completed blocks, and ETA.

## Features

- Keeps the original EPUB's chapter structure and most of its source formatting.
- Places original and translated text in **equal-width 50/50 columns**, like two books open side by side.
- Duplicates original EPUB images in the corresponding right-hand column, including standalone illustrations.
- Prompts Qwen for a close, structure-conscious translation instead of polished literary prose.
- Checks whether the source text matches the language you entered before translating.
- Defaults to English as the target language.
- Writes a new file named `originalname_translated.epub`; the source file is not overwritten.
- Caches repeated text blocks during a run to avoid translating identical passages twice.
- Counts paragraphs/text blocks before translating and shows a live progress bar plus the current chapter.
- Keeps the model loaded between requests and bounds generation length to reduce avoidable delays.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running locally
- A Qwen model downloaded into Ollama
- An EPUB file that contains readable XHTML/HTML text

## 1. Install Ollama and download Qwen

Install Ollama from its official site, then pull a Qwen model. For example:

```bash
ollama pull qwen2.5:7b
```

Make sure the local Ollama service is running. The default API endpoint is `http://localhost:11434`.

LinguaLoom automatically selects the newest Qwen generation installed locally; it does not ask you to choose a model. To use a newer generation, pull it with Ollama before running LinguaLoom.

## 2. Set up LinguaLoom

Unzip the repository, open a terminal in the project directory, and create a virtual environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## 3. Run it

```bash
lingualoom
```

Or, without installing the console command:

```bash
python -m lingualoom
```

The program will ask you to:

1. Choose the original `.epub` file with a file picker (`BROWSE` by default), or enter a filename, relative path, or absolute path.
2. Enter the original language (for example, `Portuguese`).
3. Choose the target language (defaults to `English`).
4. Choose an installed Qwen model.

LinguaLoom samples the EPUB and asks Qwen to verify the original language. If the model says the sample may not match, you can cancel or explicitly continue.

The result is saved beside the original file. For example:

```text
The Little Prince.epub
The Little Prince_translated.epub
```

## Translation approach

LinguaLoom instructs Qwen to preserve meaning, sentence order, and source structure, favouring transparent, close translations over elegant rewrites. It does not ask for grammar explanations or translator notes.

Translations are generated in small batches of up to five paragraphs per model request to reduce overhead. If Qwen returns malformed or incomplete batch output, LinguaLoom retries that batch paragraph by paragraph. Translation results are cached individually, so a failed run can reuse successful translations. Translations are rendered block by block. A block may be a paragraph, heading, list item, caption, or definition-list item. EPUBs vary widely, so complex layouts, footnotes, tables, poetry, and unusual markup may not be represented perfectly. Always proofread the output before relying on it for study.

## Privacy

The model inference requests go to your local Ollama server at `localhost`. LinguaLoom does not intentionally send EPUB text to a cloud translation service. Ollama itself may have separate software-update or telemetry behaviour depending on its configuration.

## Troubleshooting

**“Could not reach Ollama”**
- Start Ollama and retry.
- Check that `http://localhost:11434/api/tags` is reachable.
- If Ollama uses a different local host or port, update `OLLAMA_URL` in `lingualoom/cli.py`.

**“No Qwen model found”**
- Run `ollama pull qwen2.5:7b`, then restart LinguaLoom.

**Slow translation**
- LinguaLoom automatically uses the newest Qwen generation already installed. Smaller model variants generally run faster; larger variants may produce better translations but need more RAM/VRAM. Pull a preferred newer Qwen generation with Ollama if it is not installed yet.
- The first run can be slower while Ollama loads the model.

**Language check fails**
- Try a more specific language name, such as `Brazilian Portuguese` or `European Portuguese`.
- Short, mixed-language, scanned, or image-only EPUBs may not contain enough text for a reliable check.

## Development

```bash
python -m pip install -e .
python -m lingualoom
```

Contributions and issue reports are welcome. Please do not commit copyrighted EPUBs or other private reading material to the repository.

## License

MIT. See [`LICENSE`](LICENSE).
