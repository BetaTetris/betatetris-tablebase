#include "python.h"

inline int ParsePieceID(PyObject* obj) {
  if (PyUnicode_Check(obj)) {
    if (PyUnicode_GET_LENGTH(obj) < 1) {
      PyErr_SetString(PyExc_KeyError, "Invalid piece symbol.");
      return -1;
    }
    switch (PyUnicode_READ_CHAR(obj, 0)) {
      case 'T': return 0;
      case 'J': return 1;
      case 'Z': return 2;
      case 'O': return 3;
      case 'S': return 4;
      case 'L': return 5;
      case 'I': return 6;
      default: {
        PyErr_SetString(PyExc_KeyError, "Invalid piece symbol.");
        return -1;
      }
    }
  } else if (PyLong_Check(obj)) {
    long x = PyLong_AsLong(obj);
    if (x < 0 || x >= 7) {
      PyErr_SetString(PyExc_IndexError, "Piece ID out of range.");
      return -1;
    }
    return x;
  } else {
    PyErr_SetString(PyExc_TypeError, "Invalid type for piece.");
    return -1;
  }
}
