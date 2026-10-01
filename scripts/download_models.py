"""Download the two official GGUFs at one pinned revision; verify SHA256."""
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Qwen/Qwen3-8B-GGUF'
NAMES = ['Qwen3-8B-Q4_K_M.gguf', 'Qwen3-8B-Q8_0.gguf']


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    folder = ROOT / 'models'
    folder.mkdir(exist_ok=True)
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        response = requests.get(f'https://huggingface.co/api/models/{REPO}?blobs=true', timeout=60)
        response.raise_for_status()
        metadata = response.json()
        files = {f['rfilename']: f for f in metadata['siblings']}
        manifest = {'repo': REPO, 'revision': metadata['sha'], 'files': {}}
        for name in NAMES:
            entry = files[name]['lfs']
            manifest['files'][name] = {'sha256': entry['sha256'], 'size': entry['size']}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for name, info in manifest['files'].items():
        target = folder / name
        if target.exists():
            if digest(target) != info['sha256']:
                raise RuntimeError(f'Existing file has wrong hash: {target}')
            print(f'Verified existing: {name}', flush=True)
            continue
        partial = folder / (name + '.part')
        for attempt in range(5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == info['size']:
                    break
                url = f"https://huggingface.co/{REPO}/resolve/{manifest['revision']}/{name}?download=true"
                with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {}, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        raise RuntimeError('Server did not honor resume range')
                    last_report = time.monotonic()
                    with partial.open('ab' if offset else 'wb') as f:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            f.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last_report > 20:
                                print(f'{name}: {offset / 1e9:.2f}/{info["size"] / 1e9:.2f} GB', flush=True)
                                last_report = time.monotonic()
                break
            except requests.RequestException as exc:
                print(f'Retry {attempt + 1}: {type(exc).__name__}', flush=True)
                if attempt == 4:
                    raise
                time.sleep(2)
        if partial.stat().st_size != info['size'] or digest(partial) != info['sha256']:
            raise RuntimeError(f'Incomplete or corrupt download: {partial}')
        partial.rename(target)
        print(f'Downloaded and SHA256 verified: {name}', flush=True)


if __name__ == '__main__':
    main()
