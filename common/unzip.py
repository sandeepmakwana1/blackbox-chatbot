import os
import zipfile
import tarfile
import gzip
from pathlib import Path
from typing import List, Dict, Any

# Assuming these are defined and available globally in your application
ARCHIVE_EXTENSIONS = {".zip", ".zipx", ".tar", ".tar.gz", ".tgz", ".gz"}


def extract_archive_sync(
    archive_path: Path, file_extension: str
) -> List[Dict[str, Any]]:
    """
    Handles archive extraction. This function performs blocking disk I/O and
    MUST be run in a separate thread pool (e.g., via asyncio.to_thread).
    """
    extracted_files_info = []
    extract_dir = archive_path.parent

    # --- Security Check: Extraction Limit ---
    MAX_EXTRACTED_FILES = 1000
    extracted_count = 0

    try:
        if file_extension in {".zip", ".zipx"}:
            with zipfile.ZipFile(archive_path, "r") as zf:
                if len(zf.namelist()) > MAX_EXTRACTED_FILES:
                    raise ValueError("Archive exceeds file count limit.")

                zf.extractall(extract_dir)

        elif file_extension in {".tar", ".tar.gz", ".tgz"}:
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(extract_dir)

        # --- COMMON LOGIC FOR ZIP/TAR EXTRACTION ---
        if file_extension in {".zip", ".zipx", ".tar", ".tar.gz", ".tgz"}:
            # Use glob to iterate over all extracted files/folders
            for extracted_file_path in extract_dir.rglob("*"):
                # Exclude the original archive file
                if extracted_file_path == archive_path:
                    continue

                # Only process actual files and stop if limit is hit
                if (
                    extracted_file_path.is_file()
                    and not extracted_file_path.is_symlink()
                ):
                    if extracted_count >= MAX_EXTRACTED_FILES:
                        raise ValueError(
                            "Archive extraction exceeded file count limit."
                        )

                    # Get the path relative to the extraction directory (e.g., 'folder/file.txt')
                    relative_filename = str(
                        extracted_file_path.relative_to(extract_dir)
                    )

                    # Skip system files, path traversal, and empty names
                    if (
                        not relative_filename.startswith(
                            (".", "__MACOSX/", ".git/", "..")
                        )
                        and relative_filename
                    ):
                        with open(extracted_file_path, "rb") as mf:
                            extracted_files_info.append(
                                {
                                    "name": relative_filename,
                                    "contents": mf.read(),
                                    "content_type": "application/octet-stream",
                                }
                            )
                        extracted_count += 1

        elif file_extension == ".gz":
            decompressed_filename = Path(archive_path.name).stem
            with gzip.open(archive_path, "rb") as gz:
                decompressed_contents = gz.read()
                extracted_files_info.append(
                    {
                        "name": decompressed_filename,
                        "contents": decompressed_contents,
                        "content_type": "application/octet-stream",
                    }
                )

    except Exception:
        raise

    return extracted_files_info
