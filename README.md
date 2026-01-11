# Content Vector

A CLI tool to ingest video content and ask questions using RAG (Retrieval-Augmented Generation).

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Set up API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Ingest videos
cvector ingest /path/to/your/videos

# 4. Ask questions
cvector ask "What are the key trading strategies mentioned?"
```

## Commands

### `ingest` - Process videos into the knowledge base

```bash
cvector ingest /path/to/videos [OPTIONS]

Options:
  -m, --model [tiny|base|small|medium|large]  Whisper model size
  -l, --language TEXT                          Language code (auto-detected if not set)
  --chunk-size INTEGER                         Chunk size in characters (default: 1000)
  --overlap INTEGER                            Overlap between chunks (default: 200)
  --db-path TEXT                               Vector database path (default: .chroma_db)
```

Model recommendations:
- `tiny` - Fastest, least accurate (good for testing)
- `base` - Good balance (default)
- `medium` - Better accuracy
- `large` - Best accuracy (requires more RAM/VRAM)

### `ask` - Query your knowledge base

```bash
cvector ask "Your question here" [OPTIONS]

Options:
  -p, --provider [openai|anthropic]  LLM provider (default: anthropic)
  -m, --model TEXT                   Specific model to use
  -n, --results INTEGER              Number of context chunks (default: 5)
  --db-path TEXT                     Vector database path
```

### `chat` - Interactive chat mode

```bash
cvector chat [OPTIONS]

Options:
  -p, --provider [openai|anthropic]  LLM provider (default: anthropic)
  --db-path TEXT                     Vector database path
```

### `stats` - View database statistics

```bash
cvector stats
```

### `clear` - Clear the database

```bash
cvector clear
```

## Supported Formats

Video: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`
Audio: `.mp3`, `.wav`, `.m4a`, `.flac`

## Requirements

- Python 3.10+
- FFmpeg (for audio extraction from video)
- At least one API key (OpenAI or Anthropic)

### Installing FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
choco install ffmpeg
```
