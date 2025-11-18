# face_guard/app.py
import sys, json, base64
from PyQt5 import QtWidgets
from pathlib import Path
from .dialogs import CreateMasterDialog, PasswordDialog, show_message
# اضافه کردن load_encrypted_file به ایمپورت‌ها برای تست بی سر و صدا
from .crypto import gen_salt, derive_key_from_password, load_encrypted_file
from .utils import DATA_FOLDER
from .storage import ProfileStore 

SETTINGS_FILE = DATA_FOLDER / 'settings.json'

def write_settings(salt: bytes):
    settings = {'master_salt': base64.b64encode(salt).decode('utf-8')}
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def gui_init_or_load_settings(app: QtWidgets.QApplication):
    """
    Handle master-password GUI flow entirely via dialogs (no console).
    Returns derived key (bytes) or None if user cancelled or invalid.
    """
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
        # کاربر دیالوگ را بست (کنسل کرد)
        return None
        
    pw = dlg.get_password()
    if not pw:
        return None
    
    key = derive_key_from_password(pw, salt)
    
    # --- اعتبارسنجی کلید (Validation) ---
    store = ProfileStore(key)
    admins = store.list_admins()
    
    # اگر هیچ ادمینی نیست، نمی‌توانیم رمز را چک کنیم، پس فرض می‌کنیم درست است
    if not admins:
        return key

    ok = False
    for a in admins:
        folder = DATA_FOLDER / a
        enc_path = folder / 'encodings.npz'
        
        if enc_path.exists():
            try:
                # تلاش برای رمزگشایی مستقیم و بی‌سروصدا
                # ما از store.load_encodings_all استفاده نمی‌کنیم چون آن تابع ارور چاپ می‌کند
                load_encrypted_file(enc_path, key)
                
                # اگر به اینجا رسیدیم یعنی رمز درست بوده و فایل باز شده
                ok = True
                break 
            except Exception:
                # رمز اشتباه است. هیچ چیزی چاپ نمی‌کنیم (Silent Fail)
                continue

    if not ok:
        # نمایش پیام خطا به کاربر (این پنجره باز می‌شود و کاربر OK می‌زند)
        show_message(None, 'Denied', 'Master password incorrect.', QtWidgets.QMessageBox.Warning)
        return None

    return key

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # تلاش برای گرفتن کلید و احراز هویت
    key = gui_init_or_load_settings(app)
    
    # اگر کلید None بود (رمز اشتباه، کنسل، یا هر خطای دیگر)
    if key is None:
        # برنامه را فوراً و کامل می‌بندیم
        sys.exit(0)
    
    # اگر موفق بود، برنامه اصلی را باز کن
    from .ui import MainWindow
    win = MainWindow(key)
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()