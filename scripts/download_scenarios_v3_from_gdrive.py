#!/usr/bin/env python
"""
Download raw scenario JSON files from Google Drive into outputs/scenarios_v3.

Default remote path:
    benchmark/outputs/scenarios_v3

Default filename pattern:
    scenario_*.json

Authentication options:
1. OAuth installed-app flow:
   python scripts/download_scenarios_v3_from_gdrive.py ^
       --credentials path\\to\\client_secret.json

2. Service account:
   python scripts/download_scenarios_v3_from_gdrive.py ^
       --service-account path\\to\\service_account.json

The script stores the OAuth refresh token locally so subsequent runs do not
need to open the browser again.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVE_PATH = "benchmark/outputs/scenarios_v3"
DEFAULT_LOCAL_DIR = REPO_ROOT / "outputs" / "scenarios_v3"
DEFAULT_TOKEN_PATH = REPO_ROOT / ".cache" / "gdrive_scenarios_v3_token.json"
DEFAULT_PATTERN = "scenario_*.json"


def _import_google_dependencies():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google.oauth2.service_account import Credentials as ServiceAccountCredentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise SystemExit(
            "Missing Google Drive dependencies. Install them with:\n"
            "  pip install google-api-python-client google-auth-httplib2 "
            "google-auth-oauthlib"
        ) from exc
    return {
        "Request": Request,
        "Credentials": Credentials,
        "ServiceAccountCredentials": ServiceAccountCredentials,
        "InstalledAppFlow": InstalledAppFlow,
        "build": build,
        "HttpError": HttpError,
        "MediaIoBaseDownload": MediaIoBaseDownload,
    }


def _escape_drive_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _iter_drive_files(service, query: str, fields: str) -> Iterable[Dict[str, Any]]:
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields=f"nextPageToken, files({fields})",
                pageToken=page_token,
                pageSize=1000,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for item in response.get("files", []):
            yield item
        page_token = response.get("nextPageToken")
        if not page_token:
            break


def _build_drive_service(args: argparse.Namespace):
    deps = _import_google_dependencies()
    Request = deps["Request"]
    Credentials = deps["Credentials"]
    ServiceAccountCredentials = deps["ServiceAccountCredentials"]
    InstalledAppFlow = deps["InstalledAppFlow"]
    build = deps["build"]

    if args.service_account:
        creds = ServiceAccountCredentials.from_service_account_file(
            str(args.service_account),
            scopes=SCOPES,
        )
    else:
        if args.credentials is None:
            raise SystemExit(
                "Provide either --credentials for OAuth or --service-account."
            )
        token_path = args.token
        creds = None
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds is not None and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if creds is None or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(args.credentials),
                SCOPES,
            )
            creds = flow.run_local_server(
                host=args.oauth_host,
                port=args.oauth_port,
                open_browser=not args.no_browser,
            )
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _resolve_drive_folder(service, drive_path: str) -> Dict[str, Any]:
    current_parent = "root"
    current_meta = {"id": "root", "name": "My Drive"}

    parts = [part for part in Path(drive_path).parts if part not in ("", ".")]
    for part in parts:
        name = _escape_drive_query(part)
        query = (
            f"'{current_parent}' in parents and trashed = false and "
            f"mimeType = '{FOLDER_MIME_TYPE}' and name = '{name}'"
        )
        matches = list(
            _iter_drive_files(
                service,
                query,
                "id, name, createdTime, modifiedTime",
            )
        )
        if not matches:
            raise FileNotFoundError(
                f"Google Drive folder not found: {drive_path!r} "
                f"(missing segment {part!r})"
            )
        if len(matches) > 1:
            options = ", ".join(f"{item['name']}[{item['id']}]" for item in matches[:5])
            raise RuntimeError(
                f"Ambiguous Google Drive folder segment {part!r} under "
                f"{current_meta['name']!r}. Candidates: {options}. "
                "Use a more specific folder or temporarily remove duplicates."
            )
        current_meta = matches[0]
        current_parent = current_meta["id"]

    return current_meta


def _list_matching_files(
    service,
    folder_id: str,
    pattern: str,
) -> List[Dict[str, Any]]:
    query = (
        f"'{folder_id}' in parents and trashed = false and "
        f"mimeType != '{FOLDER_MIME_TYPE}'"
    )
    items = list(
        _iter_drive_files(
            service,
            query,
            "id, name, size, md5Checksum, modifiedTime",
        )
    )
    matched = [item for item in items if fnmatch.fnmatch(item["name"], pattern)]
    matched.sort(key=lambda item: item["name"])
    return matched


def _should_skip_download(remote: Dict[str, Any], local_path: Path, overwrite: bool) -> bool:
    if overwrite or not local_path.exists():
        return False
    remote_size = remote.get("size")
    if remote_size is None:
        return True
    try:
        return local_path.stat().st_size == int(remote_size)
    except OSError:
        return False


def _download_file(service, downloader_cls, file_id: str, destination: Path) -> None:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.FileIO(destination, mode="wb")
    downloader = downloader_cls(buffer, request, chunksize=1024 * 1024)
    done = False
    try:
        while not done:
            _status, done = downloader.next_chunk()
    finally:
        buffer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw scenario JSONs from Google Drive into outputs/scenarios_v3.",
    )
    parser.add_argument(
        "--drive-path",
        default=DEFAULT_DRIVE_PATH,
        help=f"Google Drive folder path under My Drive. Default: {DEFAULT_DRIVE_PATH}",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Filename glob within the Drive folder. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Destination folder. Default: {DEFAULT_LOCAL_DIR}",
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=None,
        help="OAuth client secrets JSON for Google installed-app auth.",
    )
    parser.add_argument(
        "--service-account",
        type=Path,
        default=None,
        help="Service account JSON with access to the Drive folder.",
    )
    parser.add_argument(
        "--token",
        type=Path,
        default=DEFAULT_TOKEN_PATH,
        help=f"Where to cache the OAuth token. Default: {DEFAULT_TOKEN_PATH}",
    )
    parser.add_argument(
        "--oauth-host",
        default="localhost",
        help="Host used for OAuth installed-app callback. Default: localhost",
    )
    parser.add_argument(
        "--oauth-port",
        type=int,
        default=0,
        help="Port used for OAuth installed-app callback. Default: 0 (auto)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser during OAuth flow.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if a same-size local file already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only download the first N matching files. Default: 0 (no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.credentials is not None and args.service_account is not None:
        raise SystemExit("Use either --credentials or --service-account, not both.")

    service = _build_drive_service(args)
    deps = _import_google_dependencies()
    downloader_cls = deps["MediaIoBaseDownload"]

    folder = _resolve_drive_folder(service, args.drive_path)
    matches = _list_matching_files(service, folder["id"], args.pattern)
    if args.limit > 0:
        matches = matches[: args.limit]

    args.local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolved Google Drive folder: {args.drive_path} [{folder['id']}]")
    print(f"Destination: {args.local_dir}")
    print(f"Matched files: {len(matches)}")

    if not matches:
        print("No matching files found.")
        return 0

    downloaded = 0
    skipped = 0
    failed = 0

    for index, item in enumerate(matches, start=1):
        destination = args.local_dir / item["name"]
        if _should_skip_download(item, destination, args.overwrite):
            skipped += 1
            print(f"[{index}/{len(matches)}] skip {item['name']}")
            continue

        if args.dry_run:
            print(f"[{index}/{len(matches)}] would download {item['name']}")
            continue

        print(f"[{index}/{len(matches)}] download {item['name']}")
        try:
            _download_file(service, downloader_cls, item["id"], destination)
            downloaded += 1
        except Exception as exc:
            failed += 1
            if destination.exists():
                try:
                    destination.unlink()
                except OSError:
                    pass
            print(f"  failed: {exc}", file=sys.stderr)

    print(
        "Done: "
        f"downloaded={downloaded} skipped={skipped} failed={failed} "
        f"target={args.local_dir}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
