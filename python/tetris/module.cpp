#include "board.h"
#include "tetris.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL TETRIS_PY_ARRAY_SYMBOL_
#include <numpy/ndarrayobject.h>

static PyMethodDef py_tetris_module_methods[] = {
  {nullptr},
};
static PyModuleDef py_tetris_module = {
  PyModuleDef_HEAD_INIT,
  "tetris",
  "Tetris module",
  -1,
  py_tetris_module_methods,
};

PyMODINIT_FUNC PyInit_tetris() {
  import_array();
  if (PyType_Ready(&py_tetris_class) < 0 ||
      PyType_Ready(&py_board_class) < 0) return nullptr;

  PyObject *m = PyModule_Create(&py_tetris_module);
  if (m == nullptr) return nullptr;

  PyObject *all = Py_BuildValue("[s,s]", "Tetris", "Board");
  Py_INCREF(&py_tetris_class);
  Py_INCREF(&py_board_class);

  if (PyModule_AddObject(m, "Tetris", (PyObject*)&py_tetris_class) < 0 ||
      PyModule_AddObject(m, "Board", (PyObject*)&py_board_class) < 0 ||
      PyModule_AddObject(m, "__all__", all) < 0) {
    Py_DECREF(&py_tetris_class);
    Py_DECREF(&py_board_class);
    Py_DECREF(m);
    Py_CLEAR(all);
    return nullptr;
  }
  return m;
}
