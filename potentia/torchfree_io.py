"""Read torch.save() zip archives (e.g. data/*.dat) without importing torch.

torch.save writes a zip containing `<name>/data.pkl` plus raw storages under
`<name>/data/<key>`.  We unpickle data.pkl with stub classes for the torch
symbols it references and rebuild tensors as numpy arrays.  Only the subset of
torch needed for the parameter files in this repo is supported.
"""
import io
import pickle
import zipfile
import collections
import numpy as np

_DTYPES = {
    'DoubleStorage': np.float64, 'FloatStorage': np.float32, 'HalfStorage': np.float16,
    'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
    'CharStorage': np.int8, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_,
}


class _StorageType:
    def __init__(self, name):
        self.name = name
        self.dtype = _DTYPES[name]


class _LazyStorage:
    def __init__(self, dtype, key, nbytes_reader):
        self.dtype, self.key, self._read = dtype, key, nbytes_reader

    def array(self):
        return np.frombuffer(self._read(self.key), dtype=self.dtype)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    flat = storage.array()
    size, stride = tuple(size), tuple(stride)
    if len(size) == 0:
        return np.array(flat[storage_offset])
    return np.lib.stride_tricks.as_strided(
        flat[storage_offset:], shape=size,
        strides=tuple(s * flat.itemsize for s in stride)).copy()


def _rebuild_parameter(data, requires_grad, backward_hooks, *a):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return _rebuild_tensor_v2
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return _rebuild_parameter
        if module == 'torch' and name in _DTYPES:
            return _StorageType(name)
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module == 'torch' and name == 'Size':
            return tuple
        if module.startswith('numpy') or module in ('_codecs', 'builtins', 'collections'):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'unsupported global {module}.{name}')

    def persistent_load(self, pid):
        typ, storage_type, key, location, numel = pid
        assert typ == 'storage'
        return _LazyStorage(storage_type.dtype, key,
                            lambda k: self.zf.read(f'{self.prefix}/data/{k}'))


def load(path):
    with zipfile.ZipFile(path) as zf:
        pkl = [n for n in zf.namelist() if n.endswith('data.pkl')][0]
        prefix = pkl.rsplit('/', 1)[0]
        return _Unpickler(io.BytesIO(zf.read(pkl)), zf, prefix).load()
