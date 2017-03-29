#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "geometry.h"
 
/* Docstrings */
static char module_docstring[] =
    "This module provides an interface for calculating squared euclidean distance";
static char euclidean128_docstring[] =
    "Calculate the squared euclidean distance of two 128-dimensional vectors";
static char euclidean2d_docstring[] =
    "Calculate the squared euclidean distance of two 2-dimensional vectors";
static char euclidean3d_docstring[] =
    "Calculate the squared euclidean distance of two 3-dimensional vectors";
static char project_point_docstring[] =
    "Calculate the 3D to 2D projection using a projection matrix(3x4)";


/* Available functions */
static PyObject *pyc_squared_euclidean128(PyObject *self, PyObject *args);
static PyObject *pyc_squared_euclidean2d(PyObject *self, PyObject *args);
static PyObject *pyc_squared_euclidean3d(PyObject *self, PyObject *args);
static PyObject *pyc_project_point(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"pyc_euclidean128", pyc_squared_euclidean128, METH_VARARGS, euclidean128_docstring},
    {"pyc_euclidean2d", pyc_squared_euclidean2d, METH_VARARGS, euclidean2d_docstring},
    {"pyc_euclidean3d", pyc_squared_euclidean3d, METH_VARARGS, euclidean3d_docstring},
    {"pyc_project_point", pyc_project_point, METH_VARARGS, project_point_docstring},
    {NULL, NULL, 0, NULL}
};
 
/* Initialize the module */
PyMODINIT_FUNC initpyc_geometry(void)
{
    PyObject *m = Py_InitModule3("pyc_geometry", module_methods, module_docstring);
    if (m == NULL)
        return;
 
    /* Load `numpy` functionality. */
    import_array();
}
 
static PyObject *pyc_squared_euclidean128(PyObject *self, PyObject *args)
{
    PyArrayObject *x_obj, *y_obj;
     NpyIter *x_iter;
     NpyIter *y_iter;
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj))
        return NULL;


    /* Get the iterators for the input arrays. */
    x_iter = NpyIter_New(x_obj, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    y_iter = NpyIter_New(y_obj, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);

    /* Get the data pointers from the iterators */
    double ** x = (double **) NpyIter_GetDataPtrArray(x_iter);
    double ** y = (double **) NpyIter_GetDataPtrArray(y_iter);

    /* Compute the squared euclidean distance */
    double value = squared_euclidean128(*x, *y);

    /* Deallocation */
    NpyIter_Deallocate(x_iter);
    NpyIter_Deallocate(y_iter);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);

    return ret;
}

static PyObject *pyc_squared_euclidean2d(PyObject *self, PyObject *args)
{
    PyArrayObject *x_obj, *y_obj;
    NpyIter *x_iter;
    NpyIter *y_iter;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj))
        return NULL;

    /* Get the iterators for the input arrays. */
    x_iter = NpyIter_New(x_obj, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    y_iter = NpyIter_New(y_obj, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);

    /* Get the data pointers from the iterators */
    double ** x = (double **) NpyIter_GetDataPtrArray(x_iter);
    double ** y = (double **) NpyIter_GetDataPtrArray(y_iter);

    /* Compute the squared euclidean distance */
    double value = squared_euclidean2d(*x, *y);

    NpyIter_Deallocate(x_iter);
    NpyIter_Deallocate(y_iter);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    return ret;
}

static PyObject *pyc_squared_euclidean3d(PyObject *self, PyObject *args)
{
    PyArrayObject *x_obj, *y_obj;
     NpyIter *x_iter;
     NpyIter *y_iter;
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj))
        return NULL;


    /* Get the iterators for the input arrays. */
    x_iter = NpyIter_New(x_obj, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    y_iter = NpyIter_New(y_obj, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);

    /* Get the data pointers from the iterators */
    double ** x = (double **) NpyIter_GetDataPtrArray(x_iter);
    double ** y = (double **) NpyIter_GetDataPtrArray(y_iter);

    /* Compute the squared euclidean distance */
    double value = squared_euclidean3d(*x, *y);

    NpyIter_Deallocate(x_iter);
    NpyIter_Deallocate(y_iter);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    return ret;
}

static PyObject *pyc_project_point(PyObject *self, PyObject *args)
{
    PyArrayObject *x_obj, *y_obj;
    NpyIter *x_iter;
    NpyIter *y_iter;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj))
        return NULL;

    /* Get the iterators for the input arrays. */
    x_iter = NpyIter_New(x_obj, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    y_iter = NpyIter_New(y_obj, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);

    /* Get the data pointers from the iterators */
    double ** P = (double **) NpyIter_GetDataPtrArray(x_iter);
    double ** point = (double **) NpyIter_GetDataPtrArray(y_iter);

    /* Compute the projection from 3D to 2D */
    double num1   = *(P[0]    ) * (*(point[0])) + *(P[0] + 1) * (*(point[0] + 1)) + *(P[0] + 2) * (*(point[0] + 2)) + *(P[0] + 3);
    double num2   = *(P[0] + 4) * (*(point[0])) + *(P[0] + 5) * (*(point[0] + 1)) + *(P[0] + 6) * (*(point[0] + 2)) + *(P[0] + 7);
    double denom  = *(P[0] + 8) * (*(point[0])) + *(P[0] + 9) * (*(point[0] + 1)) + *(P[0] + 10) * (*(point[0] + 2)) + *(P[0] + 11);

    double ptx = num1 / denom;
    double pty = num2 / denom;

    /* Deallocation */
    NpyIter_Deallocate(x_iter);
    NpyIter_Deallocate(y_iter);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("dd", ptx, pty);
    return ret;
}