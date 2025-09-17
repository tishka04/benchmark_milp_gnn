import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable, Tuple

try:
    from pdfminer.high_level import extract_pages, extract_text
    from pdfminer.layout import LTFigure, LTImage
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pdfminer.six is required. Install it with `pip install pdfminer.six`."
    ) from exc


def discover_targets(raw_inputs: Iterable[str]) -> Iterable[Tuple[Path, Path]]:
    """Yield (pdf_path, base_dir) tuples for each provided input."""
    for raw in raw_inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            logging.warning("Skipping missing path %s", path)
            continue
        if path.is_dir():
            for pdf in path.rglob("*.pdf"):
                yield pdf, path
        elif path.suffix.lower() == ".pdf":
            yield path, path.parent
        else:
            logging.warning("Skipping non-PDF file %s", path)


def ensure_output_location(
    pdf_path: Path, base_dir: Path, output_root: Path, overwrite: bool
) -> Path:
    relative = pdf_path.relative_to(base_dir)
    target_dir = output_root / relative.parent / pdf_path.stem
    if overwrite and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def write_text(pdf_path: Path, target_dir: Path) -> Path:
    text_output = target_dir / "text.txt"
    text = extract_text(pdf_path)
    text_output.write_text(text or "", encoding="utf-8")
    return text_output


def iter_layout_images(layout):
    for element in layout:
        if isinstance(element, LTImage):
            yield element
        elif isinstance(element, LTFigure):
            yield from iter_layout_images(element)


def export_images(pdf_path: Path, target_dir: Path) -> int:
    images_dir = target_dir / "images"
    images_dir.mkdir(exist_ok=True)
    seen_hashes = set()
    image_counter = 0

    for page_number, page_layout in enumerate(extract_pages(pdf_path), start=1):
        for image_index, image in enumerate(iter_layout_images(page_layout), start=1):
            if not hasattr(image, "stream") or image.stream is None:
                continue
            try:
                data = image.stream.get_data()
            except Exception as exc:  # pragma: no cover
                logging.warning(
                    "Failed to read image data from %s page %s: %s", pdf_path, page_number, exc
                )
                continue
            digest = hashlib.sha1(data).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            extension = guess_extension(data)
            filename = f"page{page_number:04d}_img{image_index}.{extension}"
            (images_dir / filename).write_bytes(data)
            image_counter += 1
    if image_counter == 0:
        try:
            images_dir.rmdir()
        except OSError:
            pass
    return image_counter


def guess_extension(data: bytes) -> str:
    import imghdr

    detected = imghdr.what(None, data)
    return detected or "bin"


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description="Extract text and images from PDFs.")
    parser.add_argument("inputs", nargs="+", help="PDF files or directories to process")
    parser.add_argument(
        "-o",
        "--output",
        default="extracted",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear any existing output folders for each PDF before writing.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce log verbosity.",
    )
    args = parser.parse_args(list(argv))

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)

    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for pdf_path, base_dir in discover_targets(args.inputs):
        total_files += 1
        target_dir = ensure_output_location(pdf_path, base_dir, output_root, args.overwrite)
        logging.info("Processing %s", pdf_path)
        text_path = write_text(pdf_path, target_dir)
        images_found = export_images(pdf_path, target_dir)
        logging.info(" -> text: %s", text_path)
        if images_found:
            logging.info(" -> images: %s found", images_found)
        else:
            logging.info(" -> no images extracted")

    if total_files == 0:
        logging.warning("No PDFs processed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))