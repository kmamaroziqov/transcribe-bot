"""Resolve Linux dependencies on Windows, then fetch verified wheels in parallel."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit, unquote

import httpx
import download_model as downloader

ROOT = downloader.ROOT
WHEELS = ROOT / '.wheels'


def prepare(group, requirements):
    report = WHEELS / f'{group}-report.json'
    command = [sys.executable, '-m', 'pip', 'install', '--dry-run', '--ignore-installed',
               '--report', str(report), '--only-binary=:all:', '--python-version', '3.12',
               '--implementation', 'cp', '--abi', 'cp312', '--timeout', '120', '--retries', '5']
    for version in range(17, 42):
        command += ['--platform', f'manylinux_2_{version}_x86_64']
    command += ['--platform', 'manylinux2014_x86_64', '--platform', 'manylinux2010_x86_64', *requirements]
    subprocess.run(command, check=True)
    entries = json.loads(report.read_text(encoding='utf-8'))['install']
    def fetch(entry):
        info = entry['download_info']
        url = info['url']
        if urlsplit(url).scheme == 'file':
            return
        target = WHEELS / unquote(urlsplit(url).path.rsplit('/', 1)[-1])
        sha = info['archive_info']['hashes']['sha256']
        if target.exists():
            size = target.stat().st_size
        else:
            response = httpx.head(url, follow_redirects=True, timeout=120)
            response.raise_for_status()
            size = int(response.headers['content-length'])
        downloader.download_file(url, size, sha, target)
    downloader.WORKERS = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, entries))
    (WHEELS / f'{group}.ready').write_text('All wheels verified.\n', encoding='utf-8')
    print(f'{group}: all dependency wheels verified.', flush=True)


if __name__ == '__main__':
    WHEELS.mkdir(exist_ok=True)
    prepare('converter', [str(WHEELS/'torch-2.8.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl'),
                          'ctranslate2==4.8.2', 'transformers==4.55.4', 'hf-xet==1.6.0'])
    prepare('runtime', ['-r', str(ROOT/'telegram_srt/requirements.txt')])
