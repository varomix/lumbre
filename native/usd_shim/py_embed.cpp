// Embedded CPython for Lumbre's script editor.
//
// This lives in the USD shim rather than a library of its own for one specific
// reason: OpenUSD's core libraries reference Python symbols, and the shim
// already links libpython to satisfy them. Lumbre must drive *that*
// interpreter — loading a second libpython into a process that already has
// USD's gives two interpreter states, two GILs, and duplicate symbols.
//
// The engine ships no standard library of its own, so the interpreter is
// started against the one vendored by scripts/vendor_python.sh. Configuration
// is done here, in C, because PyConfig's layout comes from the real header;
// declaring it on the Odin side would be guesswork that breaks on a version
// bump.
//
// Everything is called from the UI thread and the GIL is never released, so
// there is no threading to reason about.

#include "usd_shim.h"

#include <Python.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void py_write_err(char* buf, int len, const char* msg) {
    if (!buf || len <= 0) return;
    std::snprintf(buf, static_cast<size_t>(len), "%s", msg ? msg : "unknown error");
}

// Appends one path to config.module_search_paths, converting to wchar_t the
// way CPython expects.
bool py_append_path(PyConfig* config, const std::string& path, char* err, int err_len) {
    wchar_t* w = Py_DecodeLocale(path.c_str(), nullptr);
    if (!w) {
        py_write_err(err, err_len, "Py_DecodeLocale failed");
        return false;
    }
    PyStatus st = PyWideStringList_Append(&config->module_search_paths, w);
    PyMem_RawFree(w);
    if (PyStatus_Exception(st)) {
        py_write_err(err, err_len, st.err_msg ? st.err_msg : "PyWideStringList_Append failed");
        return false;
    }
    return true;
}

char* py_dup(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

} // namespace

// ── the `lumbre_native` module ──────────────────────────────────────────────
//
// Rather than binding every renderer call through C, the module exposes a
// single entry point that forwards a command name and a JSON payload to the
// host and returns JSON. The pleasant API lives in a small pure-Python
// `lumbre` module on top, so extending it means touching Odin and Python, not
// this file.

static LumbreCommandFn g_command_fn = nullptr;
static void* g_command_user = nullptr;

extern "C" void lumbre_py_set_command_handler(LumbreCommandFn fn, void* user) {
    g_command_fn = fn;
    g_command_user = user;
}

static PyObject* lumbre_native_call(PyObject*, PyObject* args) {
    const char* cmd = nullptr;
    const char* payload = nullptr;
    if (!PyArg_ParseTuple(args, "ss", &cmd, &payload)) return nullptr;

    if (!g_command_fn) {
        PyErr_SetString(PyExc_RuntimeError, "lumbre: no host attached");
        return nullptr;
    }
    char* result = g_command_fn(g_command_user, cmd, payload);
    if (!result) {
        PyErr_Format(PyExc_RuntimeError, "lumbre: command failed: %s", cmd);
        return nullptr;
    }
    PyObject* out = PyUnicode_FromString(result);
    std::free(result);
    return out;
}

static PyMethodDef lumbre_native_methods[] = {
    {"call", lumbre_native_call, METH_VARARGS,
     "call(command, json_payload) -> json_result"},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef lumbre_native_module = {
    PyModuleDef_HEAD_INIT, "lumbre_native", "Lumbre host bridge", -1,
    lumbre_native_methods, nullptr, nullptr, nullptr, nullptr,
};

static PyObject* lumbre_native_init(void) {
    return PyModule_Create(&lumbre_native_module);
}

extern "C" int lumbre_py_is_initialized(void) {
    return Py_IsInitialized() ? 1 : 0;
}

extern "C" int lumbre_py_init(const char* home, char* err_buf, int err_buf_len) {
    if (Py_IsInitialized()) return 1;
    if (!home) {
        py_write_err(err_buf, err_buf_len, "lumbre_py_init: null home");
        return 0;
    }

    // Must be registered before Py_InitializeFromConfig; afterwards the
    // builtin module table is fixed.
    if (PyImport_AppendInittab("lumbre_native", lumbre_native_init) != 0) {
        py_write_err(err_buf, err_buf_len, "PyImport_AppendInittab failed");
        return 0;
    }

    PyConfig config;
    // Isolated config is the important part: it ignores PYTHONHOME,
    // PYTHONPATH and PYTHONSTARTUP, and skips site.py. Without it, anyone
    // with a conda environment active would have their stdlib picked up
    // against this engine — the exact mismatch vendoring exists to avoid,
    // and one that would only ever show up on someone else's machine.
    PyConfig_InitIsolatedConfig(&config);
    config.site_import = 0;
    config.user_site_directory = 0;
    config.install_signal_handlers = 0;

    std::string home_s(home);
    std::string zip = home_s + "/python312.zip";
    std::string dynload = home_s + "/lib-dynload";

    PyStatus st = PyConfig_SetBytesString(&config, &config.home, home);
    if (PyStatus_Exception(st)) {
        py_write_err(err_buf, err_buf_len, st.err_msg ? st.err_msg : "PyConfig_SetBytesString(home) failed");
        PyConfig_Clear(&config);
        return 0;
    }

    // Replace the computed search path entirely; the vendored layout is the
    // only thing that should ever be importable.
    config.module_search_paths_set = 1;
    if (!py_append_path(&config, zip, err_buf, err_buf_len) ||
        !py_append_path(&config, dynload, err_buf, err_buf_len) ||
        !py_append_path(&config, home_s, err_buf, err_buf_len)) {
        PyConfig_Clear(&config);
        return 0;
    }

    st = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(st)) {
        py_write_err(err_buf, err_buf_len, st.err_msg ? st.err_msg : "Py_InitializeFromConfig failed");
        return 0;
    }
    return 1;
}

extern "C" void lumbre_py_finalize(void) {
    if (Py_IsInitialized()) Py_FinalizeEx();
}

extern "C" const char* lumbre_py_version(void) {
    return Py_IsInitialized() ? Py_GetVersion() : "";
}

extern "C" char* lumbre_py_run(const char* code, int* out_ok) {
    if (out_ok) *out_ok = 0;
    if (!Py_IsInitialized() || !code) return nullptr;

    // Capture output by swapping sys.stdout/sys.stderr for a StringIO. This
    // catches tracebacks too, since PyRun_SimpleString prints them to
    // sys.stderr rather than returning them.
    PyObject* io_mod = PyImport_ImportModule("io");
    if (!io_mod) {
        PyErr_Clear();
        return py_dup("internal error: could not import io");
    }
    PyObject* buf = PyObject_CallMethod(io_mod, "StringIO", nullptr);
    Py_DECREF(io_mod);
    if (!buf) {
        PyErr_Clear();
        return py_dup("internal error: could not create StringIO");
    }

    // Borrowed references; hold them across the swap so they are not collected.
    PyObject* old_out = PySys_GetObject("stdout");
    PyObject* old_err = PySys_GetObject("stderr");
    Py_XINCREF(old_out);
    Py_XINCREF(old_err);

    PySys_SetObject("stdout", buf);
    PySys_SetObject("stderr", buf);

    int rc = PyRun_SimpleString(code);

    if (old_out) PySys_SetObject("stdout", old_out);
    if (old_err) PySys_SetObject("stderr", old_err);
    Py_XDECREF(old_out);
    Py_XDECREF(old_err);

    std::string captured;
    PyObject* value = PyObject_CallMethod(buf, "getvalue", nullptr);
    if (value) {
        const char* utf8 = PyUnicode_AsUTF8(value);
        if (utf8) captured = utf8;
        Py_DECREF(value);
    }
    PyErr_Clear();
    Py_DECREF(buf);

    if (out_ok) *out_ok = (rc == 0) ? 1 : 0;
    return py_dup(captured);
}
