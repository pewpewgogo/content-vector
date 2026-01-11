"""CLI interface for content-vector."""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Content Vector - Ask questions about your video content."""
    pass


@cli.command()
@click.argument("folder", type=click.Path(exists=True))
@click.option("--model", "-m", default="base",
              type=click.Choice(["tiny", "base", "small", "medium", "large"]),
              help="Whisper model size (larger = more accurate but slower)")
@click.option("--language", "-l", default=None,
              help="Language code (e.g., 'en'). Auto-detected if not specified.")
@click.option("--chunk-size", default=1000, help="Chunk size in characters")
@click.option("--overlap", default=200, help="Overlap between chunks")
@click.option("--db-path", default=".chroma_db", help="Path to store vector database")
def ingest(folder: str, model: str, language: str, chunk_size: int, overlap: int, db_path: str):
    """
    Ingest videos from FOLDER into the vector database.

    This will:
    1. Transcribe all video/audio files using Whisper
    2. Chunk the transcripts into smaller pieces
    3. Store them in a local ChromaDB vector database
    """
    from .transcribe import transcribe_folder, get_media_files
    from .chunker import chunk_transcripts
    from .vectordb import add_chunks, get_stats

    # Show what we're processing
    files = get_media_files(folder)
    console.print(f"\n[bold]Found {len(files)} media files to process[/bold]\n")

    for f in files:
        console.print(f"  • {f.name}")

    console.print()

    # Transcribe
    with console.status("[bold green]Transcribing videos..."):
        transcripts = transcribe_folder(folder, model_size=model, language=language)

    console.print(f"[green]✓ Transcribed {len(transcripts)} files[/green]\n")

    # Chunk
    with console.status("[bold green]Chunking transcripts..."):
        chunks = chunk_transcripts(transcripts, chunk_size=chunk_size, overlap=overlap)

    console.print(f"[green]✓ Created {len(chunks)} chunks[/green]\n")

    # Store in vector DB
    with console.status("[bold green]Storing in vector database..."):
        added = add_chunks(chunks, db_path=db_path)

    console.print(f"[green]✓ Added {added} chunks to database[/green]\n")

    # Show stats
    stats = get_stats(db_path=db_path)
    console.print(Panel(
        f"Total chunks: {stats['total_chunks']}\nSource files: {stats['source_files']}",
        title="Database Stats"
    ))


@cli.command()
@click.argument("question")
@click.option("--provider", "-p", default="anthropic",
              type=click.Choice(["openai", "anthropic"]),
              help="LLM provider to use")
@click.option("--model", "-m", default=None,
              help="Model to use (defaults to gpt-4o-mini or claude-sonnet-4-20250514)")
@click.option("--results", "-n", default=5, help="Number of context chunks to retrieve")
@click.option("--db-path", default=".chroma_db", help="Path to vector database")
def ask(question: str, provider: str, model: str, results: int, db_path: str):
    """
    Ask a QUESTION about your video content.

    Examples:
        cvector ask "What are the best stop-loss strategies?"
        cvector ask "Explain the key points about risk management" -p openai
    """
    from .query import ask as ask_question

    with console.status(f"[bold green]Searching and generating answer with {provider}..."):
        result = ask_question(
            question=question,
            provider=provider,
            n_results=results,
            db_path=db_path,
            model=model
        )

    # Display answer
    console.print()
    console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))

    # Display sources
    if result["sources"]:
        console.print(f"\n[dim]Sources ({result['context_chunks']} chunks from {len(result['sources'])} files):[/dim]")
        for source in result["sources"]:
            console.print(f"  • {source}")
    console.print()


@cli.command()
@click.option("--db-path", default=".chroma_db", help="Path to vector database")
def stats(db_path: str):
    """Show database statistics."""
    from .vectordb import get_stats

    s = get_stats(db_path=db_path)

    table = Table(title="Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks", str(s["total_chunks"]))
    table.add_row("Source Files", str(s["source_files"]))

    console.print(table)

    if s["files"]:
        console.print("\n[bold]Indexed Files:[/bold]")
        for f in s["files"]:
            console.print(f"  • {f}")


@cli.command()
@click.option("--db-path", default=".chroma_db", help="Path to vector database")
@click.confirmation_option(prompt="Are you sure you want to clear the database?")
def clear(db_path: str):
    """Clear all data from the database."""
    from .vectordb import clear_database

    clear_database(db_path=db_path)
    console.print("[green]✓ Database cleared[/green]")


@cli.command()
@click.option("--db-path", default=".chroma_db", help="Path to vector database")
@click.option("--provider", "-p", default="anthropic",
              type=click.Choice(["openai", "anthropic"]),
              help="LLM provider to use")
def chat(db_path: str, provider: str):
    """Start an interactive chat session."""
    from .query import ask as ask_question

    console.print(Panel(
        "Interactive chat mode. Type 'exit' or 'quit' to end.\n"
        f"Using {provider} for responses.",
        title="Content Vector Chat"
    ))

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ")

            if question.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if not question.strip():
                continue

            with console.status("[dim]Thinking...[/dim]"):
                result = ask_question(
                    question=question,
                    provider=provider,
                    db_path=db_path
                )

            console.print(f"\n[bold green]Assistant:[/bold green] {result['answer']}")

            if result["sources"]:
                console.print(f"\n[dim]Sources: {', '.join(result['sources'])}[/dim]")

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break


def main():
    cli()


if __name__ == "__main__":
    main()
