import sys, json, base64
from PyQt5 import QtWidgets
from pathlib import Path
from .dialogs import CreateMasterDialog, PasswordDialog
from .crypto import gen_salt, derive_key_from_password, save_encrypted_file
from .utils import DATA_FOLDER
SETTINGS_FILE = DATA_FOLDER / 'settings.json'

def write_settings(salt: bytes):
    settings = {'master_salt': base64.b64encode(salt).decode('utf-8')}
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def gui_init_or_load_settings(app: QtWidgets.QApplication):
    # Ensure data folder exists
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_FILE.exists():
        dlg = CreateMasterDialog()
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None
        pw = dlg.get_password()
        if not pw:
            return None
        salt = gen_salt()
        write_settings(salt)
        key = derive_key_from_password(pw, salt)
        # Do NOT keep raw password; just return key
        return key
    # existing settings: ask password
    with open(SETTINGS_FILE, 'r') as f:
        settings = json.load(f)
    salt = base64.b64decode(settings['master_salt'].encode('utf-8'))
    dlg = PasswordDialog()
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return None
    pw = dlg.get_password()
    if not pw:
        return None
    key = derive_key_from_password(pw, salt)
    # validate by attempting to decrypt any encodings file if exists
    from .storage import ProfileStore
    store = ProfileStore(key)
    admins = store.list_admins()
    if not admins:
        return key
    # try at least one enc file
    ok = False
    for a in admins:
        folder = DATA_FOLDER / a
        enc = folder / 'encodings.npz'
        if enc.exists():
            try:
                store.load_encodings_all()
                ok = True; break
            except Exception:
                continue
    if not ok:
        QtWidgets.QMessageBox.warning(None, 'Denied', 'Master password incorrect.')
        return None
    return key

def main():
    app = QtWidgets.QApplication(sys.argv)
    key = gui_init_or_load_settings(app)
    if key is None:
        return
    from .ui import MainWindow
    win = MainWindow(key)
    win.show()
    sys.exit(app.exec_())
