"""Download the CPU conversion wheel from PyTorch's official index in pieces."""
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, unquote
import httpx
from download_model import download_file, ROOT

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(v for k,v in attrs if k == 'href')

if __name__ == '__main__':
    index = 'https://download.pytorch.org/whl/cpu/torch/'
    response = httpx.get(index, timeout=120)
    response.raise_for_status()
    parser = Links()
    parser.feed(response.text)
    name = 'torch-2.8.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl'
    choices = [urljoin(index, link) for link in parser.links if unquote(urlsplit(link).path).endswith('/'+name)]
    if len(choices) != 1:
        raise SystemExit('Expected one matching official CPU wheel.')
    link = urlsplit(choices[0])
    sha = link.fragment.removeprefix('sha256=')
    url = choices[0].split('#')[0]
    response = httpx.head(url, follow_redirects=True, timeout=120)
    response.raise_for_status()
    size = int(response.headers['content-length'])
    print('Downloading CPU converter dependency:', name, flush=True)
    download_file(url, size, sha, ROOT / '.wheels' / name)
