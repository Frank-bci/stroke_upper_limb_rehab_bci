from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

ARTICLE_ID = "21679035"
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"


def _get_json_with_retries(url: str, retries: int = 3) -> dict:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            response = requests.get(url, timeout=60, headers={"User-Agent": "stroke-bci-mvp/0.1"})
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - retry helper should capture network variants.
            last_error = exc
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def _download_file(url: str, dest: Path, retries: int = 3) -> None:
    last_error: Exception | None = None
    tmp = dest.with_suffix(dest.suffix + ".part")
    for _ in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "stroke-bci-mvp/0.1"}) as response:
                response.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - network retries are intentionally broad.
            last_error = exc
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Download metadata or EDF zip for Figshare stroke MI dataset.")
    parser.add_argument("--out", default="data/raw/figshare_stroke")
    parser.add_argument("--metadata-only", action="store_true", help="Only download JSON/TSV/MD metadata files.")
    parser.add_argument("--include-edf-zip", action="store_true", help="Also download edffile.zip, about 463 MB.")
    parser.add_argument("--edf-only", action="store_true", help="Only download edffile.zip.")
    parser.add_argument("--strict", action="store_true", help="Fail if any individual file download fails.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    article = _get_json_with_retries(API_URL)
    (out / "figshare_article_metadata.json").write_text(
        json.dumps(article, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    wanted = []
    for file_info in article["files"]:
        name = file_info["name"]
        if args.edf_only and name == "edffile.zip":
            wanted.append(file_info)
        elif args.edf_only:
            continue
        elif name.endswith((".json", ".tsv", ".md")):
            wanted.append(file_info)
        elif args.include_edf_zip and name == "edffile.zip":
            wanted.append(file_info)

    for file_info in wanted:
        dest = out / file_info["name"]
        if dest.exists() and dest.stat().st_size == int(file_info["size"]):
            print(f"Skipping existing {dest}")
            continue
        print(f"Downloading {file_info['name']} -> {dest}")
        try:
            _download_file(file_info["download_url"], dest)
        except Exception as exc:  # noqa: BLE001 - keep metadata download useful under flaky TLS.
            if args.strict:
                raise
            print(f"WARNING: skipped {file_info['name']} after download failure: {exc}")

    if args.metadata_only and not args.include_edf_zip:
        print("Metadata downloaded. Re-run with --include-edf-zip to fetch the EDF archive.")


if __name__ == "__main__":
    main()
