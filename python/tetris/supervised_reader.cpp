#include "supervised_reader.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL TETRIS_PY_ARRAY_SYMBOL_
#include <numpy/ndarrayobject.h>

namespace {

void SupervisedDataReaderDealloc(PythonSupervisedDataReader* self) {
  self->~PythonSupervisedDataReader();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
  Py_TYPE(self)->tp_free((PyObject*)self);
#pragma GCC diagnostic pop
}

PyObject* SupervisedDataReaderNew(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PythonSupervisedDataReader* self = (PythonSupervisedDataReader*)type->tp_alloc(type, 0);
  // leave initialization to __init__
  return (PyObject*)self;
}

int SupervisedDataReaderInit(PythonSupervisedDataReader* self, PyObject* args, PyObject* kwds) {
  static const char *kwlist[] = {"filename", "seed", nullptr};
  PyObject* fname = nullptr;
  unsigned long long seed = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|K", (char**)kwlist, &fname, &seed)) {
    return -1;
  }
  if (!PyUnicode_Check(fname)) {
    PyErr_SetString(PyExc_TypeError, "Filename must be a string");
    return -1;
  }
  const char* c_str = PyUnicode_AsUTF8(fname);
  new(self) PythonSupervisedDataReader(c_str, seed);
  return 0;
}

PyObject* SupervisedDataReader_ReadBatch(PythonSupervisedDataReader* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"sz", nullptr};
  unsigned long long sz;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", (char**)kwlist, &sz)) {
    return nullptr;
  }
  auto data = self->ReadBatch(sz);
  PyObject *x = data.first.ToPython(), *y;
  npy_intp N = data.second.size();
  npy_intp dim_arr[2] = {N, (npy_intp)std::tuple_size<MultiStateY::value_type>::value};
  y = PyArray_SimpleNew(2, dim_arr, NPY_FLOAT32);
  memcpy(PyArray_DATA((PyArrayObject*)y), data.second.data(), N * sizeof(MultiStateY::value_type));
  PyObject* ret = PyTuple_Pack(2, x, y);
  Py_DECREF(x);
  Py_DECREF(y);
  return ret;
}

PyMethodDef py_supervised_data_reader_class_methods[] = {
    {"ReadBatch", (PyCFunction)SupervisedDataReader_ReadBatch, METH_VARARGS | METH_KEYWORDS,
     "Get a batch of training data"},
    {nullptr}};

} // namespace

PyTypeObject py_supervised_data_reader_class = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "tetris.SupervisedDataReader",      // tp_name
    sizeof(PythonSupervisedDataReader), // tp_basicsize
    0,                       // tp_itemsize
    (destructor)SupervisedDataReaderDealloc, // tp_dealloc
    0,                       // tp_print
    0,                       // tp_getattr
    0,                       // tp_setattr
    0,                       // tp_reserved
    0,                       // tp_repr
    0,                       // tp_as_number
    0,                       // tp_as_sequence
    0,                       // tp_as_mapping
    0,                       // tp_hash
    0,                       // tp_call
    0,                       // tp_str
    0,                       // tp_getattro
    0,                       // tp_setattro
    0,                       // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
    "SupervisedDataReader class",             // tp_doc
    0,                       // tp_traverse
    0,                       // tp_clear
    0,                       // tp_richcompare
    0,                       // tp_weaklistoffset
    0,                       // tp_iter
    0,                       // tp_iternext
    py_supervised_data_reader_class_methods, // tp_methods
    0,                       // tp_members
    0,                       // tp_getset
    0,                       // tp_base
    0,                       // tp_dict
    0,                       // tp_descr_get
    0,                       // tp_descr_set
    0,                       // tp_dictoffset
    (initproc)SupervisedDataReaderInit, // tp_init
    0,                       // tp_alloc
    SupervisedDataReaderNew, // tp_new
};
