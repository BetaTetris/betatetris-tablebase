#pragma once

#define PY_SSIZE_T_CLEAN
#if __has_include(<python3.8/Python.h>)
#include <python3.8/Python.h>
#elif __has_include(<python3.9/Python.h>)
#include <python3.9/Python.h>
#elif __has_include(<python3.10/Python.h>)
#include <python3.10/Python.h>
#elif __has_include(<python3.11/Python.h>)
#include <python3.11/Python.h>
#elif __has_include(<python3.12/Python.h>)
#include <python3.12/Python.h>
#elif __has_include(<python3.13/Python.h>)
#include <python3.13/Python.h>
#else
#include <Python.h>
#endif
