"""The embedded interpreter itself: one copy of USD, and UTF-8 text.

    ./lumbre --script python/tests/test_environment.py

Both of these failed once. `from pxr import Usd` used to load a second
libusd_tf from the OpenUSD build tree and abort (scripts/vendor_pxr.sh), and the
isolated interpreter used to run text I/O as US-ASCII.
"""

import ctypes
import json
import locale
import os
import sys
import tempfile

import lumbre
from pxr import Usd, UsdUtils

failures = []


def check(label, ok):
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


check("host is the CLI", lumbre.host() == "cli")

# Every image dyld loaded, straight from dyld. A USD dylib from anywhere but
# lib/darwin/usd means pxr's bindings resolved outside the vendored copy.
dyld = ctypes.CDLL(None)
dyld._dyld_image_count.restype = ctypes.c_uint32
dyld._dyld_get_image_name.restype = ctypes.c_char_p
dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
images = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
# libusd_shim is Lumbre's own library and lives one level up, in lib/darwin.
usd_images = [p for p in images
              if os.path.basename(p).startswith("libusd_") and os.path.basename(p) != "libusd_shim.dylib"]
outside = [p for p in usd_images if "/lib/darwin/usd/" not in p]
check(f"all {len(usd_images)} USD dylibs are the vendored ones", usd_images and not outside)
for p in outside:
    print("     loaded from outside:", p)
tf_copies = [p for p in usd_images if os.path.basename(p) == "libusd_tf.dylib"]
check("exactly one libusd_tf", len(tf_copies) == 1)

stage = Usd.Stage.CreateInMemory()
check("stage cache accepts a stage", UsdUtils.StageCache.Get().Insert(stage).IsValid())

check("UTF-8 mode is on", sys.flags.utf8_mode == 1)
check("preferred encoding is UTF-8", locale.getpreferredencoding(False).lower() in ("utf-8", "utf8"))
name = "silla_café_ñ"
print("     non-ASCII print:", name)
path = os.path.join(tempfile.mkdtemp(), "roundtrip.json")
with open(path, "w") as f:
    json.dump({"class": name}, f, ensure_ascii=False)
with open(path) as f:
    check("non-ASCII JSON round trip", json.load(f)["class"] == name)

try:
    lumbre.stats()
    check("a GUI-only command raises in the CLI", False)
except RuntimeError as e:
    check("a GUI-only command raises in the CLI, by name", "'stats'" in str(e))

if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
