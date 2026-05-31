import argparse
import json
import mimetypes
import os
import uuid
from pathlib import Path
from urllib import error, request

DEFAULT_API_BASE_URL = os.getenv('SHANK_API_URL', 'http://127.0.0.1:8088').rstrip('/')
DEFAULT_TIMEOUT_SECONDS = int(os.getenv('SHANK_API_TIMEOUT', '30'))
UPLOAD_TIMEOUT_SECONDS = int(os.getenv('SHANK_API_UPLOAD_TIMEOUT', '120'))
DEFAULT_UPLOAD_CONTENT_TYPE = 'application/octet-stream'
VALID_REQUEST_TYPES = {'upload', 'melody'}


def _base_url(api_base_url: str | None = None) -> str:
    return (api_base_url or DEFAULT_API_BASE_URL).rstrip('/')


def _request_json(
    method: str,
    path: str,
    *,
    api_base_url: str | None = None,
    payload: dict | None = None,
    raw_body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    req_headers = {'Accept': 'application/json'}
    if headers:
        req_headers.update(headers)
    data = raw_body
    if payload is not None and raw_body is not None:
        raise ValueError('Only one of payload or raw_body may be provided')
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        req_headers['Content-Type'] = 'application/json'
    req = request.Request(
        f'{_base_url(api_base_url)}{path}',
        data=data,
        headers=req_headers,
        method=method,
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode('utf-8')
    except error.HTTPError as exc:
        detail = exc.reason
        try:
            body = exc.read().decode('utf-8')
            decoded = json.loads(body)
            detail = decoded.get('detail') or decoded
        except Exception:
            pass
        raise RuntimeError(f'SHANK API {method} {path} failed: {detail}') from exc

    return json.loads(body)


def _build_multipart_upload(file_path: Path, boundary: str) -> bytes:
    content_type = mimetypes.guess_type(file_path.name)[0] or DEFAULT_UPLOAD_CONTENT_TYPE
    header = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        f'Content-Type: {content_type}\r\n\r\n'
    ).encode('utf-8')
    footer = f'\r\n--{boundary}--\r\n'.encode('utf-8')
    return b''.join([header, file_path.read_bytes(), footer])


def get_health(*, api_base_url: str | None = None) -> dict:
    return _request_json('GET', '/', api_base_url=api_base_url)


def submit_url(url: str, *, api_base_url: str | None = None) -> dict:
    return _request_json('POST', '/tasks/url', api_base_url=api_base_url, payload={'url': url})


def submit_audio_file(
    file_path: str,
    *,
    requested_type: str | None = None,
    api_base_url: str | None = None,
) -> dict:
    if requested_type not in {None, 'melody'}:
        raise ValueError("requested_type must be None or 'melody'")

    audio_path = Path(file_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f'Audio file not found: {audio_path}')

    endpoint = '/tasks/melody' if requested_type == 'melody' else '/tasks/upload'
    boundary = f'shank-mcp-{uuid.uuid4().hex}'
    body = _build_multipart_upload(audio_path, boundary)
    return _request_json(
        'POST',
        endpoint,
        api_base_url=api_base_url,
        raw_body=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        timeout=UPLOAD_TIMEOUT_SECONDS,
    )


def get_task(task_id: str, *, api_base_url: str | None = None) -> dict:
    return _request_json('GET', f'/tasks/{task_id}', api_base_url=api_base_url)


def list_completed_tasks(*, api_base_url: str | None = None) -> dict:
    return _request_json('GET', '/tasks/completed', api_base_url=api_base_url)


def list_task_artifacts(task_id: str, *, api_base_url: str | None = None) -> dict:
    return _request_json('GET', f'/tasks/{task_id}/artifacts', api_base_url=api_base_url)


def build_server(*, api_base_url: str | None = None):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP('shank')

    @mcp.tool()
    def shank_health() -> dict:
        """Return SHANK API health and service metadata."""
        return get_health(api_base_url=api_base_url)

    @mcp.tool()
    def shank_submit_url(youtube_url: str) -> dict:
        """Queue a YouTube URL task and return task_id/status."""
        return submit_url(youtube_url, api_base_url=api_base_url)

    @mcp.tool()
    def shank_submit_audio(file_path: str, requested_type: str | None = None) -> dict:
        """Upload audio from a local path; pass requested_type='melody' for melody tasks."""
        normalized_type = (requested_type or 'upload').strip().lower()
        if normalized_type not in VALID_REQUEST_TYPES:
            raise ValueError("requested_type must be 'upload' or 'melody'")
        actual_requested_type = None if normalized_type == 'upload' else normalized_type
        return submit_audio_file(file_path, requested_type=actual_requested_type, api_base_url=api_base_url)

    @mcp.tool()
    def shank_get_task(task_id: str) -> dict:
        """Fetch task status/details by task_id."""
        return get_task(task_id, api_base_url=api_base_url)

    @mcp.tool()
    def shank_list_completed_tasks() -> dict:
        """List completed SHANK tasks."""
        return list_completed_tasks(api_base_url=api_base_url)

    @mcp.tool()
    def shank_list_task_artifacts(task_id: str) -> dict:
        """List available artifact names for a task."""
        return list_task_artifacts(task_id, api_base_url=api_base_url)

    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog='api.mcp_server',
        description='Run the SHANK MCP server for automation clients.',
    )
    parser.add_argument(
        '--api-url',
        default=DEFAULT_API_BASE_URL,
        help='Base URL for the SHANK HTTP API (default: %(default)s)',
    )
    args = parser.parse_args(argv)

    server = build_server(api_base_url=args.api_url)
    server.run()


if __name__ == '__main__':
    main()
