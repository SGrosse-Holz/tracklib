"""
Parallelization for tracklib functions

Use this as context for parallel-aware functions:
>>> import tracklib as tl
... from multiprocessing import Pool
...
... with Pool(16) as mypool:
...     with tl.util.Parallelize(mypool.imap):
...         # parallel-aware stuff goes here

Note that ``Pool.imap`` is preferred over ``Pool.map``, since it gives an
iterator instead of a list, so is the more literal replacement of the built-in
``map()``. If the code in question is okay with this, you can also use
``Pool.imap_unordered``.
"""
_map = map
_umap = map

class Parallelize:
    def __init__(self, map_like, map_like_unordered=None):
        self.map = map_like
        if map_like_unordered is None:
            self.umap = map_like
        else:
            self.umap = map_like_unordered

    def __enter__(self):
        global _map, _umap
        _map = self.map
        _umap = self.umap

    def __exit__(self, type, value, traceback):
        global _map, _umap
        _map = map
        _umap = map
        return False # raise anything that might have happened
