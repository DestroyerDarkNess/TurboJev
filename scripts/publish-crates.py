"""Publish the facade and its local dependency closure, dependencies first."""
import json
import os
import re
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RETRY_STATE = ROOT / 'artifacts/crates-retry-at.txt'


def wait_for_registry():
    if RETRY_STATE.exists():
        deadline = float(RETRY_STATE.read_text())
        if deadline > time.time():
            print(f'Waiting {int(deadline - time.time())} seconds for crates.io rate limit.', flush=True)
        while time.time() < deadline:
            time.sleep(max(0, min(30, deadline - time.time())))


def main():
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--no-deps', '--format-version', '1', '--locked'], cwd=ROOT))
    packages = {p['name']: p for p in metadata['packages']}
    ordered, visited = [], set()

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        package = packages[name]
        for dep in package['dependencies']:
            if dep.get('path') and dep['name'] in packages:
                visit(dep['name'])
        ordered.append(package)

    # Include the GGUF runtime and shared native host, not only the facade.
    for root_package in ('turbojev', 'turbojev-runtime-llamacpp', 'turbojev-host-native'):
        visit(root_package)
    for package in ordered:
        name, version = package['name'], package['version']
        request = urllib.request.Request(
            f'https://crates.io/api/v1/crates/{name}/{version}',
            headers={'User-Agent': 'TurboJev local publisher (github.com/DestroyerDarkNess/TurboJev)'})
        try:
            with urllib.request.urlopen(request) as response:
                existing = json.load(response)
            if existing['version'].get('yanked'):
                raise RuntimeError(f'{name}@{version} is yanked')
            print(f'Already published: {name}@{version}', flush=True)
            continue
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        print(f'Publishing {name}@{version}', flush=True)
        while True:
            wait_for_registry()
            result = subprocess.run(
                ['cargo', 'publish', '--locked', '--allow-dirty', '-p', name],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env={**os.environ, 'CARGO_TARGET_DIR': str(ROOT / 'target/publish-verify')})
            print(result.stdout, flush=True)
            if result.returncode == 0:
                break
            retry = re.search(r'Please try again after (.+? GMT)', result.stdout)
            if '429 Too Many Requests' not in result.stdout or not retry:
                raise RuntimeError(f'Publishing {name} failed (exit {result.returncode})')
            deadline = parsedate_to_datetime(retry.group(1)).timestamp() + 5
            RETRY_STATE.parent.mkdir(parents=True, exist_ok=True)
            RETRY_STATE.write_text(str(deadline))
            print(f'Registry rate limit: resuming {name} after {retry.group(1)}.', flush=True)


if __name__ == '__main__':
    main()
