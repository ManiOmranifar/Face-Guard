from face_guard.utils import sanitize_name, atomic_write_bytes
from pathlib import Path
import os
def test_sanitize_name():
    assert sanitize_name('user!@#') == 'user'
    assert sanitize_name('') == 'user'
def test_atomic_write(tmp_path):
    p = tmp_path / 'a' / 'b.bin'
    atomic_write_bytes(p, b'hello')
    assert p.exists()
    assert p.read_bytes() == b'hello'
