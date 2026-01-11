"""Video transcription using MLX Whisper (optimized for Apple Silicon)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import mlx_whisper
from tqdm import tqdm


# Supported video/audio extensions
SUPPORTED_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3', '.wav', '.m4a', '.flac'}

# MLX Whisper model mapping
MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
}


def get_media_files(path: str) -> list[Path]:
    """Get all supported media files from a folder or single file."""
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"Path does not exist: {path}")

    # If it's a single file, return it if supported
    if file_path.is_file():
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [file_path]
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # It's a directory, find all media files
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(file_path.glob(f"*{ext}"))
        files.extend(file_path.glob(f"*{ext.upper()}"))

    return sorted(files)


def transcribe_file(
    file_path: Path,
    model_name: str,
    language: Optional[str] = None
) -> dict:
    """Transcribe a single media file using MLX Whisper."""
    result = mlx_whisper.transcribe(
        str(file_path),
        path_or_hf_repo=model_name,
        language=language,
        verbose=False
    )
    return {
        "file": file_path.name,
        "path": str(file_path),
        "text": result["text"],
        "segments": result["segments"],
        "language": result["language"]
    }


def transcribe_folder(
    folder: str,
    model_size: str = "base",
    language: Optional[str] = None,
    output_dir: Optional[str] = None
) -> list[dict]:
    """Transcribe all media files in a folder."""
    files = get_media_files(folder)

    if not files:
        raise ValueError(f"No supported media files found in {folder}")

    model_name = MODEL_MAP.get(model_size, MODEL_MAP["base"])
    print(f"Using MLX Whisper model: {model_name}")

    results = []
    for file_path in tqdm(files, desc="Transcribing"):
        try:
            result = transcribe_file(file_path, model_name, language)
            results.append(result)

            # Save individual transcript if output_dir specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                transcript_file = output_path / f"{file_path.stem}.txt"
                transcript_file.write_text(result["text"])

        except Exception as e:
            print(f"Error transcribing {file_path.name}: {e}")
            continue

    return results
