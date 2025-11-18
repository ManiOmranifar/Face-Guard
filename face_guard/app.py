# face_guard/app.py
import sys, json, base64
from PyQt5 import QtWidgets
from pathlib import Path
from .dialogs import CreateMasterDialog, PasswordDialog, show_message
from .crypto import gen_salt, derive_key_from_password, load_encrypted_file
from .utils import DATA_FOLDER
from .storage import ProfileStore 

SETTINGS_FILE = DATA_FOLDER / 'settings.json'

def write_settings(salt: bytes):
    settings = {'master_salt': base64.b64encode(salt).decode('utf-8')}
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def gui_init_or_load_settings(app: QtWidgets.QApplication):
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # --- حالت اول: تنظیمات وجود ندارد (اولین اجرا) ---
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
        return key

    # --- حالت دوم: تنظیمات وجود دارد (درخواست رمز) ---
    with open(SETTINGS_FILE, 'r') as f:
        try:
            settings = json.load(f)
        except Exception:
            show_message(None, 'Error', 'settings.json corrupted.', QtWidgets.QMessageBox.Warning)
            return None

    salt = None
    try:
        salt = base64.b64decode(settings['master_salt'].encode('utf-8'))
    except Exception:
        show_message(None, 'Error', 'Invalid settings file.', QtWidgets.QMessageBox.Warning)
        return None

    dlg = PasswordDialog()
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return None
        
    pw = dlg.get_password()
    if not pw:
        return None
    
    key = derive_key_from_password(pw, salt)
    
    store = ProfileStore(key)
    admins = store.list_admins()
    
    if not admins:
        return key

    ok = False
    for a in admins:
        folder = DATA_FOLDER / a
        enc_path = folder / 'encodings.npz'
        
        if enc_path.exists():
            try:
                load_encrypted_file(enc_path, key)
                
                ok = True
                break 
            except Exception:
                continue

    if not ok:
        show_message(None, 'Denied', 'Master password incorrect.', QtWidgets.QMessageBox.Warning)
        return None

    return key

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    key = gui_init_or_load_settings(app)
    
    if key is None:
        sys.exit(0)
    
    from .ui import MainWindow
    win = MainWindow(key)
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()