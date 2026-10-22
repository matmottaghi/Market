"""Server-side per-user Windows DPAPI credential storage; environment remains supported."""
import ctypes
from ctypes import wintypes
import os
import hashlib
from pathlib import Path

from .storage import atomic_bytes


class Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_byte))]


def _crypt(content, decrypt=False):
    if os.name != "nt":
        raise RuntimeError("Secure key persistence requires Windows DPAPI; use BRS_API_KEY on this OS")
    buffer = ctypes.create_string_buffer(content)
    source = Blob(len(content), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    result = Blob()
    library = ctypes.WinDLL("crypt32", use_last_error=True)
    if decrypt:
        success = library.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result))
    else:
        success = library.CryptProtectData(ctypes.byref(source), "Market DSS", None, None, None, 1, ctypes.byref(result))
    if not success:
        raise RuntimeError("Windows protected configuration could not be accessed")
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree(ctypes.cast(result.data, ctypes.c_void_p))


def _current_sid():
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    token = wintypes.HANDLE()
    security.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    if not security.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
        raise RuntimeError("Windows account identity unavailable")
    try:
        needed = wintypes.DWORD()
        security.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not security.GetTokenInformation(token, 1, buffer, needed, ctypes.byref(needed)):
            raise RuntimeError("Windows account identity unavailable")
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        output = wintypes.LPWSTR()
        security.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
        if not security.ConvertSidToStringSidW(sid, ctypes.byref(output)):
            raise RuntimeError("Windows account identity unavailable")
        try:
            return output.value
        finally:
            kernel.LocalFree.argtypes = [ctypes.c_void_p]
            kernel.LocalFree(ctypes.cast(output, ctypes.c_void_p))
    finally:
        kernel.CloseHandle(token)


def _restrict_folder(path):
    """Current account and SYSTEM only; no subprocess or inherited broad permissions."""
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    convert = security.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.DWORD)]
    convert.restype = wintypes.BOOL
    descriptor = ctypes.c_void_p()
    if not convert(f"D:P(A;OICI;FA;;;{_current_sid()})(A;OICI;FA;;;SY)", 1, ctypes.byref(descriptor), None):
        raise RuntimeError("Windows could not prepare protected configuration permissions")
    try:
        apply = security.SetFileSecurityW
        apply.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
        apply.restype = wintypes.BOOL
        if not apply(str(path), 0x80000004, descriptor):
            raise RuntimeError("Windows could not restrict protected configuration permissions")
    finally:
        release = ctypes.WinDLL("kernel32").LocalFree
        release.argtypes = [ctypes.c_void_p]
        release(descriptor)


class CredentialStore:
    def __init__(self, data_dir):
        self.legacy_path = Path(data_dir) / ".secrets" / "brsapi.dpapi"
        account = hashlib.sha256(_current_sid().encode()).hexdigest()[:16] if os.name == "nt" else "environment"
        self.path = Path(data_dir) / f".secrets-{account}" / "brsapi.dpapi"

    def resolve(self):
        environment = os.environ.get("BRS_API_KEY", "").strip()
        try:
            for path in (self.path, self.legacy_path):
                if path.exists():
                    key = _crypt(path.read_bytes(), decrypt=True).decode("utf-8")
                    return key, {"configured": bool(key), "source": "Windows DPAPI", "status": "AVAILABLE"}
        except (PermissionError, OSError, RuntimeError):
            # A different Windows account cannot decrypt this store. Do not break public pages.
            return environment, {"configured": bool(environment), "source": "environment" if environment else None,
                "status": "PROTECTED_STORE_UNAVAILABLE",
                "message": "کلید ذخیره‌شده برای حساب فعلی ویندوز قابل خواندن نیست؛ برنامه را با حساب ذخیره‌کننده اجرا یا کلید را دوباره در تنظیمات ذخیره کنید."}
        return environment, {"configured": bool(environment), "source": "environment" if environment else None,
                             "status": "AVAILABLE" if environment else "NOT_CONFIGURED"}

    def get(self):
        return self.resolve()[0]

    def status(self):
        return self.resolve()[1]

    def save(self, key):
        key = key.strip()
        if not key or len(key) > 1024 or any(c.isspace() for c in key):
            raise ValueError("کلید واردشده خالی یا نامعتبر است.")
        encrypted = _crypt(key.encode("utf-8"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _restrict_folder(self.path.parent)
        atomic_bytes(self.path, encrypted)
