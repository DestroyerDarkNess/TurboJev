"""Package cross-built Mach-O artifacts without claiming native execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import struct

from native_assets import architecture, node_package, targets, verify_machine

root = Path(__file__).resolve().parents[1]
target = next(t for t in targets() if t['rid'] == sys.argv[1])
output = root / 'artifacts/unix' / target['rid']
version = json.loads((root / 'bindings/node/package.json').read_text())['version']


def install_name(data):
    offset = 32
    for _ in range(struct.unpack_from('<I', data, 16)[0]):
        command, size = struct.unpack_from('<II', data, offset)
        if command == 0xD:
            start = offset + struct.unpack_from('<I', data, offset + 8)[0]
            return data[start:offset + size].split(b'\0', 1)[0].decode()
        offset += size
    return None


for directory in (output / 'native', output / 'node'):
    for library in directory.iterdir():
        if library.name.startswith('libggml-blas'):
            library.unlink()
            continue
        if library.suffix in ('.dylib', '.node'):
            identity = install_name(library.read_bytes())
            # Preserve SONAME aliases so dyld can recognize one library instance.
            if identity and not identity.startswith(('@rpath/', '@loader_path/')):
                subprocess.run(['llvm-install-name-tool-18', '-id', '@rpath/' + library.name, str(library)], check=True)
manifest = {'name':node_package(target), 'version':version, 'main':'turbojev.node',
            'os':['darwin'], 'cpu':[target['arch']], 'license':'Apache-2.0', 'engines':{'node':'>=18'}}
(output / 'node/package.json').write_text(json.dumps(manifest, indent=2)+'\n')
subprocess.run([sys.executable, 'scripts/check-native-node-package.py', str(output / 'node')], cwd=root, check=True)
subprocess.run([sys.executable, '-m', 'maturin', 'build', '--release', '--locked', '--target', target['rust'],
                '--features', 'llamacpp', '--out', str(output / 'wheels'), '--auditwheel', 'skip'],
               cwd=root / 'bindings/python', env=dict(os.environ), check=True)
for wheel in (output / 'wheels').glob('*.whl'):
    subprocess.run([sys.executable, 'scripts/bundle-native-wheel.py', str(wheel), str(output / 'native')], cwd=root, check=True)
    subprocess.run([sys.executable, 'scripts/check-native-python-wheel.py', str(wheel)], cwd=root, check=True)
    with zipfile.ZipFile(wheel) as archive:
        extension = next(name for name in archive.namelist() if name.startswith('turbojev/_native') and name.endswith('.so'))
        (output / 'node' / Path(extension).name).write_bytes(archive.read(extension))
subprocess.run([sys.executable, 'scripts/check-macho-dependencies.py', str(output / 'node')], cwd=root, check=True)
# The extension was staged beside its dependencies solely for link inspection.
for extension in (output / 'node').glob('_native*.so'):
    extension.unlink()
hashes = {}
for library in (output / 'native').iterdir():
    verify_machine(library.read_bytes(), architecture(target), library.name)
    hashes[library.name] = hashlib.sha256(library.read_bytes()).hexdigest()
evidence = {'version':version, 'rid':target['rid'], 'runtime_sha256':hashes,
            'cross_compiled':True, 'link_checks_passed':True,
            'python_real_smoke_passed':False, 'node_real_smoke_passed':False}
(output / 'build-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print('Cross-build packaged. Native macOS execution remains unverified.')
