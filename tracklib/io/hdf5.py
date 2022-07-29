import numpy as np
import h5py

_TYPE_KW = '_HDF5_ORIG_TYPE_'

# The registry registers custom read/write functions
# It is a dict, where the key is a unique string identifier for the type to be
# handled and the entry is a tuple of
# 
#   (type, writer, reader)
# 
# where signatures for the writer and reader are
# 
#   hdf5_container = writer(object, name, hdf5_base_point)
#   object = reader(hdf5_container)
#
# Note that the signature of the writer function allows to pass methods from
# user-defined classes
#
# Writer functions should use the @hdf5.writer decorator (which writes the
# registry string to an attribute, such that when reading the file we know
# which reader to use)
#
# When registering read/write capabilities for a user-defined class, we
# recommend adding some random hash to the string identifier to ensure that it
# stays unique for as long as possible. tracklib uses the hash 602354027
reader_writer_registry = {}

def registrystring(obj):
    try:
        return [key for key, (ty, _, _) in reader_writer_registry.items() if ty == type(obj)][0]
    except Exception:
        raise RuntimeError(f"Could not find {str(type(obj))} in reader_writer_registry")

def writer(func):
    def wrapper(obj, name, hdf5_base):
        container = func(obj, name, hdf5_base)
        if container:
            container.attrs[_TYPE_KW] = registrystring(obj)
        return container
    return wrapper

def write(obj, name, hdf5_base):
    try:
        my_reg = registrystring(obj)
    except RuntimeError:
        if np.isscalar(obj):
            hdf5_base.attrs[name] = obj
            return None
        else:
            raise

    _, my_write, _ = reader_writer_registry[my_reg]
    return my_write(obj, name, hdf5_base)

def read(hdf5_container):
    try:
        my_reg = hdf5_container.attrs[_TYPE_KW]
    except KeyError:
        if isinstance(hdf5_container, h5py.Group):
            my_reg = 'dict'
        else: # pragma: no cover
            raise RuntimeError(f"Do not understand format of {hdf5_container.name}")

    _, _, my_read = reader_writer_registry[my_reg]
    return my_read(hdf5_container)

###################### Library for built-ins and numpy #########################

def new_group(hdf5_base, name):
    # This can be used to write directly to the root instead of creating new
    # groups
    if name is None:
        return hdf5_base
    return hdf5_base.create_group(name)

@writer
def write_dict(obj, name, hdf5_base):
    group = new_group(hdf5_base, name)
    for key in obj:
        _ = write(obj[key], key, group)
    return group

def read_group_as_dict(group):
    out = {name : read(group[name]) for name in group}
    out.update(group.attrs)
    try:
        del out[_TYPE_KW]
    except KeyError:
        pass
    return out

reader_writer_registry['dict'] = (dict, write_dict, read_group_as_dict)

@writer
def write_ndarray(obj, name, hdf5_base):
    if len(obj.shape) > 0:
        dset = hdf5_base.create_dataset(name, data=obj)
    else:
        hdf5_base.attrs[name] = obj
        dset = None
    return dset

def read_dataset_as_ndarray(dset):
    return np.asarray(dset)

reader_writer_registry['np.ndarray'] = (np.ndarray, write_ndarray, read_dataset_as_ndarray)

@writer
def write_iterable(obj, name, hdf5_base):
    try:
        first_type = type(next(iter(obj)))
    except StopIteration:
        return new_group(hdf5_base, name)

    if (not np.dtype(first_type).kind in ['O', 'U'] # don't try to write numpy arrays for objects or strings
        and all(type(entry) == first_type for entry in obj)
        and name is not None
       ):
        container = hdf5_base.create_dataset(name, data=np.asarray(list(obj)))
    else:
        container = new_group(hdf5_base, name)
        for i, entry in enumerate(obj):
            _ = write(entry, str(i), container)
    return container

def read_iterable(hdf5_container):
    mytype, _, _ = reader_writer_registry[hdf5_container.attrs[_TYPE_KW]]
    if isinstance(hdf5_container, h5py.Dataset):
        return mytype(read_dataset_as_ndarray(hdf5_container))
    else:
        data = read_group_as_dict(hdf5_container)
        return mytype(data[key] for key in sorted(data))

reader_writer_registry['list'] = (list, write_iterable, read_iterable)
reader_writer_registry['tuple'] = (tuple, write_iterable, read_iterable)
reader_writer_registry['set'] = (set, write_iterable, read_iterable)

@writer
def write_generic_class(obj, name, hdf5_base):
    return write(vars(obj), name, hdf5_base)

def read_generic_class(hdf5_container):
    # This requires that the class can be instantiated without arguments
    cls, _, _ = reader_writer_registry[hdf5_container.attrs[_TYPE_KW]]
    obj = cls()
    obj.__dict__.update(read_group_as_dict(hdf5_container))
    return obj
