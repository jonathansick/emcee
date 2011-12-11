#include <Python.h>
#include <numpy/arrayobject.h>

/* Docstrings */
static char doc[] = "A module that computes the K-means decomposition of a dataset.\n";
static char kmeans_doc[] =
"Compute the K-means decomposition of a dataset\n\n"\
"The dataset has the shape (P, D) where P is the number of samples and\n"\
"D is the dimension of the parameter space. Also, K is the number of\n"\
"means to use.\n\n"\
"Parameters\n"\
"----------\n"\
"data : numpy.ndarray (P, D)\n"\
"    The dataset.\n\n"\
"means : numpy.ndarray (K, D)\n"\
"    The mean vectors. The data in this object will be overwritten.\n\n"\
"rs : numpy.ndarray (P,)\n"\
"    The vector of responsibilities.  Each entry is an integer indicating\n"\
"    the closest mean to a particular point in the dataset. This object\n"\
"    will also have it's data overwritten.\n\n"\
"tol : float\n"\
"    The convergence criterion for the relative change in the likelihood\n\n"\
"maxiter : int\n"\
"    The maximum number of iterations to perform.\n\n"\
"verbose : bool\n"\
"    Print convergence messages?\n\n"\
"Returns\n"\
"-------\n"\
"iter : int\n"\
"    The number of iterations performed.\n\n";

PyMODINIT_FUNC init_algorithms(void);
static PyObject *algorithms_kmeans(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"kmeans", algorithms_kmeans, METH_VARARGS, kmeans_doc},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_algorithms(void)
{
    PyObject *m = Py_InitModule3("_algorithms", module_methods, doc);
    if (m == NULL)
        return;
    import_array(); /* Load NumPy */
}

static PyObject *algorithms_kmeans(PyObject *self, PyObject *args)
{
    /* parse the input tuple */
    PyObject *data_obj = NULL, *means_obj = NULL, *rs_obj = NULL;
    double tol;
    int maxiter, verbose;
    if (!PyArg_ParseTuple(args, "OOOdii", &data_obj, &means_obj, &rs_obj, &tol, &maxiter, &verbose))
        return NULL;

    /* get numpy arrays */
    PyObject *data_array  = PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *means_array = PyArray_FROM_OTF(means_obj, NPY_DOUBLE, NPY_INOUT_ARRAY);
    PyObject *rs_array    = PyArray_FROM_OTF(rs_obj, NPY_INTP, NPY_INOUT_ARRAY);
    if (data_array == NULL || means_array == NULL || rs_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "input objects can't be converted to arrays.");
        Py_XDECREF(data_array);
        Py_XDECREF(means_array);
        Py_XDECREF(rs_array);
        return NULL;
    }

    double *data  = (double*)PyArray_DATA(data_array);
    double *means = (double*)PyArray_DATA(means_array);
    long   *rs    = (long*)PyArray_DATA(rs_array);

    int p, d, k;
    int P = (int)PyArray_DIM(data_array, 0);
    int D = (int)PyArray_DIM(data_array, 1);
    int K = (int)PyArray_DIM(means_array, 0);

    double *dists = (double*)malloc(K*sizeof(double));
    long   *N_rs  = (long*)malloc(K*sizeof(long));

    double L = 1.0;
    int iter;
    for (iter = 0; iter < maxiter; iter++) {
        double L_new = 0.0, dL;
        for (p = 0; p < P; p++) {
            double min_dist = -1.0;
            for (k = 0; k < K; k++) {
                dists[k] = 0.0;
                for (d = 0; d < D; d++) {
                    double diff = means[k*D+d] - data[p*D+d];
                    dists[k] += diff*diff;
                }
                if (min_dist < 0 || dists[k] < min_dist) {
                    min_dist = dists[k];
                    rs[p] = k;
                }
            }
            L_new += dists[rs[p]];
        }

        /* check for convergence */
        dL = fabs(L_new - L)/L;
        if (iter > 5 && dL < tol)
            break;
        else
            L = L_new;

        /* update means */
        for (k = 0; k < K; k++)
            N_rs[k] = 0;
        for (p = 0; p < P; p++) {
            N_rs[rs[p]] += 1;

            for (d = 0; d < D; d++) {
                means[rs[p]*D + d] += data[p*D + d];
            }
        }

        for (k = 0; k < K; k++) {
            for (d = 0; d < D; d++) {
                means[k*D + d] /= (double)N_rs[k];
            }
        }
    }

    if (verbose && iter < maxiter)
        printf("K-means converged after %d iterations\n", iter);
    else if (verbose)
        printf("K-means didn't converge after %d iterations\n", iter);

    /* clean up */
    Py_DECREF(data_array);
    Py_DECREF(means_array);
    Py_DECREF(rs_array);
    free(dists);
    free(N_rs);

    /* return None */
    PyObject *ret = Py_BuildValue("i", iter);
    return ret;
}

