"""Verify local models and launch one hidden llama-server per NVIDIA GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
# Path to the llama-server binary of llama.cpp build b10299; override with --executable.
DEFAULT_EXE = Path('llama-server.exe')


def main(executable):
    manifest = json.loads((ROOT / 'models/manifest.json').read_text())
    runtime = ROOT / 'results' / ('servers_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    runtime.mkdir()
    version = subprocess.check_output([str(executable), '--version'], text=True, stderr=subprocess.STDOUT)
    devices = subprocess.check_output([str(executable), '--list-devices'], text=True, stderr=subprocess.STDOUT)
    (runtime / 'version.txt').write_text(version + '\n' + devices)
    if 'CUDA0' not in devices or 'CUDA1' not in devices:
        raise RuntimeError('Both CUDA devices must be available')
    launches = []
    for index, label in enumerate(['Q4_K_M', 'Q8_0']):
        model = ROOT / 'models' / f'Qwen3-8B-{label}.gguf'
        expected = manifest['files'][model.name]
        print(f'Verifying {model.name} ...', flush=True)
        with model.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if digest != expected['sha256']:
            raise RuntimeError(f'Hash mismatch: {model}')
        port = 8080 + index
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} is already occupied; inspect existing server before starting')
        command = [str(executable), '-m', str(model), '--device', f'CUDA{index}',
                   '--split-mode', 'none', '-ngl', '99', '-c', str(config.SERVER_CONTEXT), '-np', '1',
                   '--host', '127.0.0.1', '--port', str(port), '--alias', model.name,
                   '--no-context-shift', '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16']
        launches.append({'label': label, 'command': command, 'model_sha256': digest, 'port': port})
    processes = []
    try:
        for launch in launches:
            label = launch['label']
            with (runtime / f'{label}.stdout.log').open('w') as stdout, (runtime / f'{label}.stderr.log').open('w') as stderr:
                process = subprocess.Popen(launch['command'], cwd=ROOT, stdout=stdout, stderr=stderr,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
            processes.append(process)
            launch['pid'] = process.pid
            (runtime / 'launches.json').write_text(json.dumps(launches, indent=2))
        deadline = time.monotonic() + 300
        for process, launch in zip(processes, launches):
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: {launch['label']}; see {runtime}")
                try:
                    response = requests.get(f"http://127.0.0.1:{launch['port']}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"Ready: {launch['label']} port {launch['port']}, pid {process.pid}", flush=True)
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError('Model loading exceeded five minutes')
        (ROOT / 'results/active_servers.json').write_text(json.dumps({'runtime': str(runtime), 'servers': launches}, indent=2))
    except BaseException:
        # Only processes created by this invocation are stopped on startup failure.
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
        raise
    print(f'Server logs: {runtime}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, default=DEFAULT_EXE)
    main(parser.parse_args().executable)
