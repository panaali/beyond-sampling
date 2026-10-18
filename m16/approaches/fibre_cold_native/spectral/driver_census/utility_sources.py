def _as_int(x, desc):
    """
    Like `operator.index`, but emits a custom exception when passed an
    incorrect type

    Parameters
    ----------
    x : int-like
        Value to interpret as an integer
    desc : str
        description to include in any error message

    Raises
    ------
    TypeError : if x is a float or non-numeric
    """
    try:
        return operator.index(x)
    except TypeError as e:
        raise TypeError(f"{desc} must be an integer, received {x}") from e

def as_series(alist, trim=True):
    """
    Return argument as a list of 1-d arrays.

    The returned list contains array(s) of dtype double, complex double, or
    object.  A 1-d argument of shape ``(N,)`` is parsed into ``N`` arrays of
    size one; a 2-d argument of shape ``(M,N)`` is parsed into ``M`` arrays
    of size ``N`` (i.e., is "parsed by row"); and a higher dimensional array
    raises a Value Error if it is not first reshaped into either a 1-d or 2-d
    array.

    Parameters
    ----------
    alist : array_like
        A 1- or 2-d array_like
    trim : boolean, optional
        When True, trailing zeros are removed from the inputs.
        When False, the inputs are passed through intact.

    Returns
    -------
    [a1, a2,...] : list of 1-D arrays
        A copy of the input data as a list of 1-d arrays.

    Raises
    ------
    ValueError
        Raised when `as_series` cannot convert its input to 1-d arrays, or at
        least one of the resulting arrays is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polyutils as pu
    >>> a = np.arange(4)
    >>> pu.as_series(a)
    [array([0.]), array([1.]), array([2.]), array([3.])]
    >>> b = np.arange(6).reshape((2,3))
    >>> pu.as_series(b)
    [array([0., 1., 2.]), array([3., 4., 5.])]

    >>> pu.as_series((1, np.arange(3), np.arange(2, dtype=np.float16)))
    [array([1.]), array([0., 1., 2.]), array([0., 1.])]

    >>> pu.as_series([2, [1.1, 0.]])
    [array([2.]), array([1.1])]

    >>> pu.as_series([2, [1.1, 0.]], trim=False)
    [array([2.]), array([1.1, 0. ])]

    """
    arrays = [np.array(a, ndmin=1, copy=None) for a in alist]
    for a in arrays:
        if a.size == 0:
            raise ValueError("Coefficient array is empty")
        if a.ndim != 1:
            raise ValueError("Coefficient array is not 1-d")
    if trim:
        arrays = [trimseq(a) for a in arrays]

    try:
        dtype = np.common_type(*arrays)
    except Exception as e:
        object_dtype = np.dtypes.ObjectDType()
        has_one_object_type = False
        ret = []
        for a in arrays:
            if a.dtype != object_dtype:
                tmp = np.empty(len(a), dtype=object_dtype)
                tmp[:] = a[:]
                ret.append(tmp)
            else:
                has_one_object_type = True
                ret.append(a.copy())
        if not has_one_object_type:
            raise ValueError("Coefficient arrays have no common type") from e
    else:
        ret = [np.array(a, copy=True, dtype=dtype) for a in arrays]
    return ret

def trimseq(seq):
    """Remove small Poly series coefficients.

    Parameters
    ----------
    seq : sequence
        Sequence of Poly series coefficients.

    Returns
    -------
    series : sequence
        Subsequence with trailing zeros removed. If the resulting sequence
        would be empty, return the first element. The returned sequence may
        or may not be a view.

    Notes
    -----
    Do not lose the type info if the sequence contains unknown objects.

    """
    if len(seq) == 0 or seq[-1] != 0:
        return seq
    else:
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != 0:
                break
        return seq[:i + 1]

@array_function_dispatch(_common_type_dispatcher)
def common_type(*arrays):
    """
    Return a scalar type which is common to the input arrays.

    The return type will always be an inexact (i.e. floating point) scalar
    type, even if all the arrays are integer arrays. If one of the inputs is
    an integer array, the minimum precision type that is returned is a
    64-bit floating point dtype.

    All input arrays except int64 and uint64 can be safely cast to the
    returned dtype without loss of information.

    Parameters
    ----------
    array1, array2, ... : ndarrays
        Input arrays.

    Returns
    -------
    out : data type code
        Data type code.

    See Also
    --------
    dtype, mintypecode

    Examples
    --------
    >>> np.common_type(np.arange(2, dtype=np.float32))
    <class 'numpy.float32'>
    >>> np.common_type(np.arange(2, dtype=np.float32), np.arange(2))
    <class 'numpy.float64'>
    >>> np.common_type(np.arange(4), np.array([45, 6.j]), np.array([45.0]))
    <class 'numpy.complex128'>

    """
    is_complex = False
    precision = 0
    for a in arrays:
        t = a.dtype.type
        if iscomplexobj(a):
            is_complex = True
        if issubclass(t, _nx.integer):
            p = 2  # array_precision[_nx.double]
        else:
            p = array_precision.get(t)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]
