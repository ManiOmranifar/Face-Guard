from PyQt5 import QtWidgets, QtCore, QtGui

class PasswordDialog(QtWidgets.QDialog):
    def __init__(self, prompt='Enter master password:', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Master Password')
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.resize(420, 140)
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(prompt)
        label.setStyleSheet('color: black;')
        layout.addWidget(label)
        self.pw_edit = QtWidgets.QLineEdit()
        self.pw_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.pw_edit.setPlaceholderText('Enter master password')
        layout.addWidget(self.pw_edit)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        self.setLayout(layout)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def get_password(self):
        return self.pw_edit.text().strip()

class CreateMasterDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Set Master Password')
        self.resize(420, 180)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Create a master password (min 4 chars)'))
        self.pw1 = QtWidgets.QLineEdit(); self.pw1.setEchoMode(QtWidgets.QLineEdit.Password)
        self.pw2 = QtWidgets.QLineEdit(); self.pw2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.pw1.setPlaceholderText('Password'); self.pw2.setPlaceholderText('Confirm password')
        layout.addWidget(self.pw1); layout.addWidget(self.pw2)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        self.setLayout(layout)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        self._ok = False

    def _on_accept(self):
        a = self.pw1.text().strip(); b = self.pw2.text().strip()
        if len(a) < 4:
            QtWidgets.QMessageBox.warning(self, 'Invalid', 'Password too short (min 4).')
            return
        if a != b:
            QtWidgets.QMessageBox.warning(self, 'Invalid', 'Passwords do not match.')
            return
        self._ok = True
        self.accept()

    def get_password(self):
        return self.pw1.text().strip() if self._ok else ''
