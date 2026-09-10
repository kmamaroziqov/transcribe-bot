"""Optional parallel downloads of the large public Linux GPU wheels."""
from pathlib import Path
import httpx
from download_model import download_file, ROOT

PACKAGES = [('nvidia-cublas-cu12', '12.8.4.1'), ('nvidia-cudnn-cu12', '9.8.0.87'), ('nvidia-cuda-runtime-cu12', '12.8.90')]

if __name__ == '__main__':
    for package, version in PACKAGES:
        response = httpx.get(f'https://pypi.org/pypi/{package}/{version}/json', timeout=60)
        response.raise_for_status()
        wheels = [f for f in response.json()['urls'] if 'manylinux' in f['filename'] and 'x86_64' in f['filename']]
        if len(wheels) != 1:
            raise SystemExit(f'Expected one Linux x86_64 wheel for {package}.')
        wheel = wheels[0]
        print(f'Downloading {wheel["filename"]}', flush=True)
        download_file(wheel['url'], wheel['size'], wheel['digests']['sha256'], ROOT / '.wheels' / wheel['filename'])
    print('All GPU wheels verified.', flush=True)
