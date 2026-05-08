import urllib.request
import zipfile
from pathlib import Path


def ensure_dataset(archive_url, archive_path, extract_dir, required_paths):
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    required_paths = [Path(path) for path in required_paths]

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print("Downloading dataset from GitHub Release:", archive_url)
        urllib.request.urlretrieve(archive_url, archive_path)
    else:
        print("Dataset archive already cached:", archive_path)

    size_mb = round(archive_path.stat().st_size / 1024 / 1024, 1)
    print("Dataset archive size, MB:", size_mb)

    if not all(path.exists() for path in required_paths):
        print(f"Extracting dataset to {extract_dir} ...")
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_dir)
    else:
        print("Dataset already extracted in this runtime.")
