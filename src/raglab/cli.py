from pathlib import Path

import chromadb
import typer

from raglab.ingestion.chunkers import recursive_chunk
from raglab.ingestion.embedder import embed_batch
from raglab.ingestion.loaders import load_markdown, load_pdf

COLLECTION_NAME = "raglab"

app = typer.Typer()


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=".chroma")
    return client.get_or_create_collection(name=COLLECTION_NAME)


@app.command()
def ingest(
    corpus_path: str = typer.Argument(
        ..., help="Path to directory containing documents"
    ),
) -> None:
    """
    Ingest all PDF and Markdown files from a directory into ChromaDB.
    """
    path = Path(corpus_path)

    if not path.exists():
        typer.echo(f"Path does not exist: {corpus_path}")
        raise typer.Exit(code=1)

    files = list(path.glob("**/*.pdf")) + list(path.glob("**/*.md"))

    if not files:
        typer.echo("No PDF or Markdown files found.")
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(files)} files — starting ingestion...")

    collection = get_chroma_collection()
    total_chunks = 0

    for file in files:
        typer.echo(f"Processing: {file.name}")

        if file.suffix == ".pdf":
            pages = load_pdf(str(file))
        else:
            pages = load_markdown(str(file))

        for page in pages:
            chunks = recursive_chunk(page["text"])

            if not chunks:
                continue

            vectors = embed_batch(chunks)

            ids = [
                f"{page['source']}::page{page['page']}::chunk{i}"
                for i in range(len(chunks))
            ]

            metadatas = [
                {"source": page["source"], "page": page["page"], "chunk_index": i}
                for i in range(len(chunks))
            ]

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=vectors,
                metadatas=metadatas,
            )

            total_chunks += len(chunks)

    typer.echo(f"Done. {total_chunks} chunks stored in ChromaDB.")


if __name__ == "__main__":
    app()
