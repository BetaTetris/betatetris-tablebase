#include "board.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL TETRIS_PY_ARRAY_SYMBOL_
#include <numpy/ndarrayobject.h>

namespace {

void BoardDealloc(PythonBoard* self) {
  self->~PythonBoard();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
  Py_TYPE(self)->tp_free((PyObject*)self);
#pragma GCC diagnostic pop
}

PyObject* BoardNew(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PythonBoard* self = (PythonBoard*)type->tp_alloc(type, 0);
  // leave initialization to __init__
  return (PyObject*)self;
}

int BoardInit(PythonBoard* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"board", nullptr};
  PyObject* obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", (char**)kwlist, &obj)) {
    return -1;
  }
  if (!obj) {
    new(self) PythonBoard(Board::Ones);
  } else if (PyUnicode_Check(obj)) {
    const char* c_str = PyUnicode_AsUTF8(obj);
    new(self) PythonBoard(std::string_view(c_str));
  } else if (PyBytes_Check(obj)) {
    if (PyBytes_Size(obj) != 25) {
      PyErr_SetString(PyExc_IndexError, "Bytes length != 25");
      return -1;
    }
    new(self) PythonBoard(reinterpret_cast<const uint8_t*>(PyBytes_AsString(obj)));
  } else if (PyArray_Check(obj)) {
    PyArrayObject *arr = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_UINT8, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (!arr) return -1;
    ByteBoard b;
    if (PyArray_NDIM(arr) != 2 || PyArray_DIM(arr, 0) != (int)b.size() || PyArray_DIM(arr, 1) != (int)b[0].size())  {
      PyErr_SetString(PyExc_IndexError, "Array shape must be (20, 10)");
      return -1;
    }
    memcpy(b.data(), PyArray_DATA(arr), sizeof(b));
    for (auto& i : b) {
      for (auto& j : i) j = j ? 1 : 0;
    }
    Py_DECREF(arr);
    new(self) PythonBoard(b);
  }
  return 0;
}

PyObject* Board_IsClean(PythonBoard* self, PyObject* Py_UNUSED(ignored)) {
  return PyBool_FromLong((long)self->IsClean());
}

PyObject* Board_IsCleanForPerfect(PythonBoard* self, PyObject* Py_UNUSED(ignored)) {
  return PyBool_FromLong((long)self->IsCleanForPerfect());
}

PyObject* Board_Count(PythonBoard* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->board.Count());
}

PyObject* Board_Height(PythonBoard* self, PyObject* Py_UNUSED(ignored)) {
  return PyLong_FromLong(self->board.Height());
}

PyObject* Board_GetBytes(PyObject* obj) {
  CompactBoard board = reinterpret_cast<PythonBoard*>(obj)->board.ToBytes();
  return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(board.data()), board.size());
}

PyObject* Board_str(PyObject* obj) {
  return PyUnicode_FromString(reinterpret_cast<PythonBoard*>(obj)->board.ToString().c_str());
}

PyMethodDef py_board_class_methods[] = {
    {"IsClean", (PyCFunction)Board_IsClean, METH_NOARGS, "Check board cleanness"},
    {"IsCleanForPerfect", (PyCFunction)Board_IsCleanForPerfect, METH_NOARGS, "Check board cleanness"},
    {"Count", (PyCFunction)Board_Count, METH_NOARGS, "Get cell count"},
    {"Height", (PyCFunction)Board_Height, METH_NOARGS, "Get board height"},
    {"GetBytes", (PyCFunction)Board_GetBytes, METH_NOARGS, "Get 25-byte representation"},
    {nullptr}};
} // namespace

PyTypeObject py_board_class = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "tetris.Board",          // tp_name
    sizeof(PythonBoard),     // tp_basicsize
    0,                       // tp_itemsize
    (destructor)BoardDealloc, // tp_dealloc
    0,                       // tp_print
    0,                       // tp_getattr
    0,                       // tp_setattr
    0,                       // tp_reserved
    Board_str,               // tp_repr
    0,                       // tp_as_number
    0,                       // tp_as_sequence
    0,                       // tp_as_mapping
    0,                       // tp_hash
    0,                       // tp_call
    Board_str,               // tp_str
    0,                       // tp_getattro
    0,                       // tp_setattro
    0,                       // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "Board class",           // tp_doc
    0,                       // tp_traverse
    0,                       // tp_clear
    0,                       // tp_richcompare
    0,                       // tp_weaklistoffset
    0,                       // tp_iter
    0,                       // tp_iternext
    py_board_class_methods,  // tp_methods
    0,                       // tp_members
    0,                       // tp_getset
    0,                       // tp_base
    0,                       // tp_dict
    0,                       // tp_descr_get
    0,                       // tp_descr_set
    0,                       // tp_dictoffset
    (initproc)BoardInit,     // tp_init
    0,                       // tp_alloc
    BoardNew,                // tp_new
};
