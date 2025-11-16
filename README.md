# Face Guard — Production-oriented Desktop App (Exercise)

**Face Guard** is a refactored, production-oriented educational desktop application for local face recognition-based access control.  
This repository is designed to be dropped into a GitHub repo and iterated on. It focuses on safer local storage, better threading, improved performance, packaging friendliness, and developer ergonomics.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Security / Limitations](#security--limitations)
- [Requirements](#requirements)
- [Install (development)](#install-development)
- [Run (development)](#run-development)
- [Project layout](#project-layout)
- [Packaging (PyInstaller)](#packaging-pyinstaller)
- [Testing & CI](#testing--ci)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
This app demonstrates a desktop pattern for:
- Capturing faces via webcam and generating face encodings (using `face_recognition`).
- Storing images, encodings and metadata encrypted on disk (Fernet). Files are written atomically and stored inside the repository root in `face_guard_data/`.
- Running camera processing in a `QThread` to keep the GUI responsive.
- Vectorized matching of encodings for performance and scalability.
- A simple GUI (PyQt5) for enrolling admins, managing profiles, and starting the monitoring loop.

This repository is explicitly built as an educational/experimental project — not a turnkey production security product. See **Security / Limitations** below for details.

---

## Features
- GUI-only master password flow (no console prompts).
- Encrypted storage of all user data using Fernet (derived from a password via PBKDF2-HMAC-SHA256).
- Encodings saved as compressed numpy `.npz` before encryption for compactness and speed.
- Atomic writes for encrypted files to avoid partial writes/corruption.
- Safe deletes via `shutil.rmtree` for admin removal.
- Camera processing in `QThread` with a vectorized matching backend for speed.
- Basic unit tests for utility functions.
- `pyinstaller.spec` and packaging notes included.
- `logging.conf` provided as an example for file logging inside `face_guard_data`.

---

## Security / Limitations
- **This project is for learning and local use only.** Do **not** deploy this as-is in critical production environments without an independent security audit.
- Secrets (password-derived keys) are kept in memory as necessary. The app avoids storing the raw password on disk, but memory zeroing is not implemented.
- For real production usage consider integration with a key management system (KMS / HSM), secure enclave, and hardened OS-level protections.
- The face recognition model used (`face_recognition` library, dlib-based) is convenient for prototyping but may not meet accuracy or anti-spoofing requirements for security-critical use-cases.
- All data is stored under the `face_guard_data/` folder in the project root by design (so nothing leaks outside the project directory).

---

## Requirements
- Python 3.10+ recommended
- System dependencies: C++ build toolchain (for `dlib` used by `face_recognition`) on Linux/Windows
- Camera available for capture
- Installable via `pip install -r requirements.txt` (file included).

`requirements.txt` includes:
```
PyQt5
opencv-python
numpy
face_recognition
cryptography
pytest
```

---

## Install (development)
1. Clone or extract repository into a folder.
2. Create and activate a virtual environment:
```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```
3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

> If `face_recognition` and `dlib` fail to build on your system, consult the library docs for platform-specific installation notes (prebuilt wheels, conda packages, or OS package manager dependencies).

---

## Run (development)
Start the app in development:
```bash
python run.py
```

Behavior on first run:
- The app will ask you (via GUI) to create a master password. That password is used to derive an encryption key for storage.
- After creating the master password, you can enroll admins (capture images + encodings) and start monitoring.

Data location: `face_guard_data/` inside the project root. All files written by the app are inside that folder.

---

## Project layout
```
face_guard_project/
├── face_guard/                 # Package code
│   ├── __init__.py
│   ├── app.py                  # Entry, master password GUI flow
│   ├── camera_worker.py        # QThread-based camera logic + vectorized matching
│   ├── crypto.py               # KDF + Fernet wrappers + encrypted save/load
│   ├── dialogs.py              # PyQt dialogs (passwords, create master)
│   ├── storage.py              # ProfileStore: create/delete/list admins
│   ├── ui.py                   # MainWindow, EnrollDialog, ManagementDialog
│   └── utils.py                # helpers (atomic write, sanitize, crop)
├── face_guard_data/            # runtime data (created automatically)
├── run.py                      # entrypoint
├── requirements.txt
├── pyinstaller.spec
├── README.md
├── README_PACKAGING.md
└── tests/                      # pytest tests for utility functions
```

---

## Packaging (PyInstaller)
A minimal `pyinstaller.spec` is included. For a single-file build example:
```bash
pip install pyinstaller
pyinstaller --onefile run.py --name face_guard --noconsole
```
After building, copy/bundle the `face_guard_data/` folder next to the generated executable — the application expects its data directory to exist at runtime.

---

## Testing & CI
A simple GitHub Actions workflow is included (`.github/workflows/ci.yml`) to run `pytest` on push and pull requests. The workflow uses Python 3.11. It will run quickly for the unit tests included (no camera access required).

Local tests:
```bash
pytest -q
```

---

## Contributing
- Fork the repo and create a feature branch.
- Write tests for new behavior when possible.
- Open a pull request and describe the change; CI will run the tests automatically.

---

## License
This project includes an MIT license (`LICENSE` file). You are free to reuse and modify the code.

---

If you want more changes (detailed logging, CI matrix, or a PR-style patch file), tell me which one and I'll generate it next.
