## Packaging with PyInstaller (local)
1. Create a virtualenv and install requirements:
   python -m venv venv
   venv\Scripts\activate (Windows) or source venv/bin/activate (Linux)
   pip install -r requirements.txt
2. Test the app: python run.py
3. Build (example):
   pyinstaller --onefile run.py --name face_guard --noconsole
4. After building, bundle the `face_guard_data` directory alongside the executable.
