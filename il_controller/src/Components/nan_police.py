"""A set of utilities designed to assist with handling and debugging NaNs in
pytorch.

* `isnan()` is an element-wise NaN check for pytorch Tensors.
* `hasnan()` determines whether or not a pytorch Tensor has any NaNs.
* `findnan()` will attempt to find NaNs in arbitrarily nested Python objects.

This module also offers a wrapped version of the entire `torch` module which
checks for NaNs in all inputs and outputs from all `torch` functions. For
example,

```
In [1]: from kindling.nan_police import torch

In [2]: x = torch.ones(2) / 0 * 0

In [3]: x
Out[3]:

nan
nan
[torch.FloatTensor of size 2]

In [4]: torch.sum(x)
---------------------------------------------------------------------------
Exception                                 Traceback (most recent call last)
<ipython-input-4-e7f45fec8fb4> in <module>()
----> 1 torch.sum(x)

~/Development/kindling/kindling/nan_police.py in __call__(self, *args, **kwargs)
    147         if argnan_path == []:
    148           raise Exception(
--> 149             f'Found a NaN at positional argument {i + 1} (of {len(args)}) when '
    150             f'calling `{path}`!'
    151           )

Exception: Found a NaN at positional argument 1 (of 1) when calling `torch.sum`!
```

Just import `from kindling.nan_police import torch` in all of your modules when
you'd like to test for NaNs! Note that this won't inject itself into all
references to torch, only those where you've imported this wrapped version of
pytorch. That means that it won't affect the behavior of torch in third-party
modules for example.
"""

import math
import numbers

import numpy as np
import torch as realtorch


def isnan(tensor):
    # Gross: https://github.com/pytorch/pytorch/issues/4767
    return tensor != tensor


def hasnan(tensor):
    return isnan(tensor).any()


def findnan(obj):
    """Find a NaN in the haystack of `obj` and return the path to finding it.
    Returns None if no NaNs could be found. Works with Python numbers, lists,
    tuples, Pytorch Tensors, Variables, and numpy arrays."""
    # Python numbers
    if isinstance(obj, numbers.Number):
        return [] if math.isnan(obj) else None

    # Pytorch Tensors
    elif isinstance(obj, realtorch.Tensor):
        return [] if hasnan(obj) else None

    # Pytorch Variables
    elif isinstance(obj, realtorch.autograd.Variable):
        if hasnan(obj.data):
            return [('Variable.data',)]
        else:
            gradnan_path = findnan(obj.grad)
            if gradnan_path == None:
                return None
            else:
                return [('Variable.grad',)] + gradnan_path

    # numpy array (See https://stackoverflow.com/questions/36783921/valueerror-when-checking-if-variable-is-none-or-numpy-array)
    elif isinstance(obj, np.ndarray):
        return [] if np.isnan(obj).any() else None

    # Python list
    elif isinstance(obj, list):
        for ix, elem in enumerate(obj):
            elemnan_path = findnan(elem)
            if elemnan_path != None:
                return [('list[]', ix)] + elemnan_path

        # Haven't found any NaNs yet, we're done
        return None

    # Python tuple
    elif isinstance(obj, tuple):
        for ix, elem in enumerate(obj):
            elemnan_path = findnan(elem)
            if elemnan_path != None:
                return [('tuple[]', ix)] + elemnan_path

        # Haven't found any NaNs yet, we're done
        return None

    # Python dict
    elif isinstance(obj, dict):
        for k, v in obj.items():
            elemnan_path = findnan(v)
            if elemnan_path != None:
                return [('dict[]', k)] + elemnan_path

        # Haven't found any NaNs yet, we're done
        return None

    # Don't know what we're looking at, assume it doesn't have NaNs.
    else:
        return None


def isinf(tensor):
    # Gross: https://github.com/pytorch/pytorch/issues/4767
    return tensor == float('inf')

def isneginf(tensor):
    # Gross: https://github.com/pytorch/pytorch/issues/4767
    return tensor == float('-inf')


def hasinf(tensor):
    return isinf(tensor).any() or isneginf(tensor).any()


def findinf(obj):
    """Find a inf in the haystack of `obj` and return the path to finding it.
    Returns None if no infs could be found. Works with Python numbers, lists,
    tuples, Pytorch Tensors, Variables, and numpy arrays."""
    # Python numbers
    if isinstance(obj, numbers.Number):
        return [] if math.isinf(obj) else None

    # Pytorch Tensors
    elif isinstance(obj, realtorch.Tensor):
        return [] if hasinf(obj) else None

    # Pytorch Variables
    elif isinstance(obj, realtorch.autograd.Variable):
        if hasinf(obj.data):
            return [('Variable.data',)]
        else:
            gradinf_path = findinf(obj.grad)
            if gradinf_path == None:
                return None
            else:
                return [('Variable.grad',)] + gradinf_path

    # numpy array (See https://stackoverflow.com/questions/36783921/valueerror-when-checking-if-variable-is-none-or-numpy-array)
    elif isinstance(obj, np.ndarray):
        return [] if np.isinf(obj).any() else None

    # Python list
    elif isinstance(obj, list):
        for ix, elem in enumerate(obj):
            eleminf_path = findinf(elem)
            if eleminf_path != None:
                return [('list[]', ix)] + eleminf_path

        # Haven't found any infs yet, we're done
        return None

    # Python tuple
    elif isinstance(obj, tuple):
        for ix, elem in enumerate(obj):
            eleminf_path = findinf(elem)
            if eleminf_path != None:
                return [('tuple[]', ix)] + eleminf_path

        # Haven't found any infs yet, we're done
        return None

    # Python dict
    elif isinstance(obj, dict):
        for k, v in obj.items():
            eleminf_path = findinf(v)
            if eleminf_path != None:
                return [('dict[]', k)] + eleminf_path

        # Haven't found any infs yet, we're done
        return None

    # Don't know what we're looking at, assume it doesn't have infs.
    else:
        return None

def path_to_string(path):
    """Convert a path found with `findnan` to a pseudo-code selector."""
    if len(path) == 0:
        return ''
    else:
        first = path[0]
        rest = path[1:]
        typ = first[0]
        if typ == 'Variable.data':
            return '.data' + path_to_string(rest)
        elif typ == 'Variable.grad':
            return '.grad' + path_to_string(rest)
        elif (typ == 'list[]') or (typ == 'tuple[]') or (typ == 'dict[]'):
            return '[' + str(first[1]) + ']' + path_to_string(rest)
        else:
            raise Exception('Unrecognized path type, ' + str(typ) + '')


class Mock(object):
    """Emulate a module or function and wrap all calls with NaN checking. Throws
    an Exception when any NaNs are found."""

    def __init__(self, obj, path):
        self._obj = obj
        self._path = path

    def __call__(self, *args, **kwargs):
        path = '.'.join(self._path)

        # check args
        for i, arg in enumerate(args):
            argnan_path = findnan(arg)
            if argnan_path != None:
                if argnan_path == []:
                    raise Exception(
                        'Found a NaN at positional argument' + str(i + 1) + '(of ' + str(len(args)) + ') when '
                        'calling `' + str(path) + '`!'
                    )
                else:
                    raise Exception(
                        'Found NaN in positional argument ' + str(i + 1) + ' (of ' + str(len(args)) + ') when '
                        'calling `' + str(path) + '`! Specifically at '
                        '`<arg>' + str(path_to_string(argnan_path)) + '`.'
                    )

        # check kwargs
        for k, v in kwargs.items():
            vnan_path = findnan(v)
            if vnan_path != None:
                if vnan_path == []:
                    raise Exception(
                        'Found a NaN at keyword argument `' + str(k) + '` when calling `' + str(path) + '`!'
                    )
                else:
                    raise Exception(
                        'Found NaN in keyword argument `' + str(k) + '` when calling `' + str(path) + '`! '
                        'Specifically at `<' + str(k) + '>' + str(path_to_string(vnan_path)) + '`.'
                    )

        result = self._obj(*args, **kwargs)

        # check result for NaNs
        resultnan_path = findnan(result)
        if resultnan_path != None:
            if resultnan_path == []:
                raise Exception('Found NaN in output from `' + str(path) + '`!')
            else:
                raise Exception(
                    'Found NaN in output from `' + str(path) + '`! Specifically at '
                    '`<out>' + str(path_to_string(resultnan_path)) + '`.'
                )

        return result

    def __getattr__(self, name):
        return Mock(getattr(self._obj, name), self._path + [name])

    def __dir__(self):
        return self._obj.__dir__()


class MockInf(object):
    """Emulate a module or function and wrap all calls with inf checking. Throws
    an Exception when any infs are found."""

    def __init__(self, obj, path):
        self._obj = obj
        self._path = path

    def __call__(self, *args, **kwargs):
        path = '.'.join(self._path)

        # check args
        for i, arg in enumerate(args):
            arginf_path = findinf(arg)
            if arginf_path != None:
                if arginf_path == []:
                    raise Exception(
                        'Found a inf at positional argument' + str(i + 1) + '(of ' + str(len(args)) + ') when '
                        'calling `' + str(path) + '`!'
                    )
                else:
                    raise Exception(
                        'Found inf in positional argument ' + str(i + 1) + ' (of ' + str(len(args)) + ') when '
                        'calling `' + str(path) + '`! Specifically at '
                        '`<arg>' + str(path_to_string(arginf_path)) + '`.'
                    )

        # check kwargs
        for k, v in kwargs.items():
            vinf_path = findinf(v)
            if vinf_path != None:
                if vinf_path == []:
                    raise Exception(
                        'Found a inf at keyword argument `' + str(k) + '` when calling `' + str(path) + '`!'
                    )
                else:
                    raise Exception(
                        'Found inf in keyword argument `' + str(k) + '` when calling `' + str(path) + '`! '
                        'Specifically at `<' + str(k) + '>' + str(path_to_string(vinf_path)) + '`.'
                    )

        result = self._obj(*args, **kwargs)

        # check result for infs
        resultinf_path = findinf(result)
        if resultinf_path != None:
            if resultinf_path == []:
                raise Exception('Found inf in output from `' + str(path) + '`!')
            else:
                raise Exception(
                    'Found inf in output from `' + str(path) + '`! Specifically at '
                    '`<out>' + str(path_to_string(resultinf_path)) + '`.'
                )

        return result

    def __getattr__(self, name):
        return MockInf(getattr(self._obj, name), self._path + [name])

    def __dir__(self):
        return self._obj.__dir__()


# torch = Mock(realtorch, ['torch'])
