from copy import deepcopy

class TaggedSet():
    """
    A set with tags for each object.

    This class can be used as an iterator in constructs such as
    ```
    for datum in myset: ...
    ```
    The subset to be iterated over can be adjusted using makeSelection(). If
    you also need the tags for each datum, use myset(giveTags=True).

    For processing data (in the sense of "applying a function to all of them")
    it is often useful to simply use the built-in map(). If the function in
    question has to overwrite the data, you can use filter(). Note that
    in-place modification can be achieved with map() though.

    Parameters
    ----------
    iterable : iterable of data or (datum, tags) pairs, optional
        most useful things to use here: an actual list, or a generator.
        Remember to set `hasTags` accordingly.
    hasTags : bool, optional
        whether the iterable in the first argument gives data only or (datum,
        tags) pairs.

    Operators
    ---------
    For a TaggedSet `myset`, the following operations are defined:
    len(myset)
        return number of data in current selection
    myset(giveTags)
        return a generator yielding single data or (datum, tags) pairs,
        depending on the value of `giveTags` (False by default).
    iter(myset)
        shortcut for (giveTags=False). This enables the construction
        ```
        for datum in myset: ...
        ```
    myset[i]
        element access within the current selection. You can use indices from 0
        to len(myset)-1.
    myset &= otherset
        add the data in `otherset` into `myset`. This is a shortcut for
        myset.mergein(otherset).
    """

    ### Utility ###

    @staticmethod
    def makeTagsSet(tags):
        """
        An input processing function making sure that the frequently used
        'tags' argument is a set of strings. Mostly for internal use.

        Parameters
        ----------
        tags : str, list of str, or set of str

        Returns
        -------
        set of str
        """
        if isinstance(tags, str):
            tags = {tags}
        elif isinstance(tags, list):
            tags = set(tags)
        elif not isinstance(tags, set):
            raise ValueError("Did not understand type of 'tags' ({})".format(str(type(tags))))

        return tags

    ### Setting up ###

    def __init__(self, iterable=[], hasTags=True):
        self._data = []
        self._tags = []
        self._selected = []

        if hasTags:
            for datum, tags in iterable:
                self.add(datum, tags)
        else:
            for datum in iterable:
                self.add(datum)

    def add(self, datum, tags=set()):
        """
        Append the given datum with the given tags.

        Parameters
        ----------
        datum : object
            the datum to append
        tags : str, list of str, or set of str, optional
            the tags to attach to this new datum

        Notes
        -----
        The newly added datum will be part of the current selection.
        """
        tags = self.makeTagsSet(tags)

        self._data.append(datum)
        self._tags.append(tags)
        self._selected.append(True)

    ### Basic usage ###

    def __iter__(self):
        """
        Iterate over all data in current selection. Simply a shortcut for
        self(), i.e. the call syntax.
        """
        return self()

    def __call__(self, giveTags=False):
        """
        Iterate through the current selection of the list.

        Parameters
        ----------
        giveTags : bool, optional
            whether to yield only data or (datum, tags) pairs.

        Yields
        ------
        datum
            the data in the current selection
        set of str, optional
            the tags associated with the datum
        """
        for (datum, tags, selected) in zip(self._data, self._tags, self._selected):
            if selected:
                if giveTags:
                    yield (datum, tags)
                else:
                    yield datum

    def __len__(self):
        """
        Give number of data in current selection
        """
        return sum(self._selected)

    def __getitem__(self, ind):
        """
        Give n-th item in current selection
        """
        for i, datum in enumerate(self):
            if i >= ind:
                return datum

    ### Selection system ###

    def makeSelection(self, **kwargs):
        """
        Mark a subset of the current set as 'current selection'. For most
        purposes, the set will behave as if it contained only these data.

        There are multiple ways to select data. Which one is used depends on
        the kwargs given. With increasing preference, these methods are
         - select by tag: use the kwargs 'tags' and 'logic'
         - select with a user-specified function: use kwarg 'selector'

        Call this without arguments to reset the selection to the whole
        dataset.

        Parameters
        ----------
        tags : str, list of str, or set of str
            the tags to select. How these go together will be determined by
            'logic'.
        logic : callable with signature bool = logic(<list of bool>)
            the logic for handling multiple tags. Set this to (the built-in)
            `all` to select the data being tagged with all the given tags, or to
            `any` (the default) to select the data having any of the given tags.
        selector : callable with signature bool = selector(datum, tags)
            should expect the datum and a set of tags as input and return True
            if the datum is to be selected, False otherwise.

        Other Parameters
        ----------------
        refining : bool, optional
            set to True to apply the current selection scheme only to those
            data that are already part of the selection. This can be used to
            successively refine a selection by using different selection
            methods (e.g. first by tag, then by some other criterion).

        See also
        --------
        refineSelection, saveSelection, restoreSelection
        """
        assert len(self._data) == len(self._tags) == len(self._selected)
        if not 'refining' in kwargs.keys():
            kwargs['refining'] = False

        # Define selector function according to given arguments
        if 'selector' in kwargs.keys():
            selector = kwargs['selector']
        elif 'tags' in kwargs.keys():
            kwargs['tags'] = TaggedSet.makeTagsSet(kwargs['tags'])
            if not 'logic' in kwargs.keys():
                kwargs['logic'] = any
            def selector(datum, tags):
                return kwargs['logic']([tag in tags for tag in kwargs['tags']])
        else:
            def selector(datum, tags):
                return True

        for i, (datum, tags, selected) in enumerate(zip(self._data, self._tags, self._selected)):
            if selected or not kwargs['refining']:
                self._selected[i] = selector(datum, tags)

    def refineSelection(self, *args, **kwargs):
        """
        A wrapper for makeSelection(..., refining=True). See that docstring.
        """
        kwargs['refining'] = True
        self.makeSelection(*args, **kwargs)

    def saveSelection(self):
        """
        Return a copy of the current selection for reference.

        This is useful when planning to undo subsequent selections without
        having to redo the whole selection from scratch.

        See also
        --------
        restoreSelection, makeSelection
        """
        return deepcopy(self._selection)

    def restoreSelection(self, selection):
        """
        Restore a previously saved selection.

        Parameters
        ----------
        selection : list of bool
            the selection that was saved using saveSelection(). Will be copied.

        See also
        --------
        saveSelection, makeSelection
        """
        self._selection = deepcopy(selection)

    def copySelection(self):
        """
        Generate a new (copied) list from the current selection.

        Returns
        -------
        The new list

        See also
        --------
        makeSelection
        """
        def gen():
            for traj, tags in self(giveTags=True):
                yield(deepcopy(traj), deepcopy(tags))

        return TaggedSet(gen())

    ### Managing multiple TaggedSets ###

    def mergein(self, other, additionalTags=set()):
        """
        Add the contents of the TaggedSet 'other' to the caller.

        Parameters
        ----------
        other : TaggedSet
            the list whose data to add
        additionalTags : str, list of str, or set of str
            additional tag(s) to add to all of the new data.

        Notes
        -----
        This can also be invoked as self &= other. In that case no additional
        tags can be added.
        """
        if not issubclass(type(other), TaggedSet):
            raise TypeError("Can only merge a TaggedSet")

        additionalTags = TaggedSet.makeTagsSet(additionalTags)
        newTags = [tags | additionalTags for tags in other._tags]

        self._data += other._data
        self._tags += newTags
        self._selected += other._selected

    def __iand__(self, other):
        """
        Shortcut for self.mergein(other). See there.
        """
        self.mergein(other)
        return self

    ### Global tag management ###

    def addTags(self, tags):
        """
        Add new tag(s) to all data in the current selection

        Parameters
        ----------
        tags : str, list of str, or set of str
            the tag(s) to add
        """
        tags = TaggedSet.makeTagsSet(tags)

        for _, curtags in self(giveTags=True):
            curtags |= tags

    def tagset(self):
        """
        Return the set of all tags in the current selection.

        Returns
        -------
        Set of all tags in the current selection
        """
        tagset = set()
        for _, tags in self(giveTags=True):
            tagset |= tags

        return tagset

        ### Probably unnecessary ###

#     def isHomogeneous(self, dtype=None, allowSubclass=False):
#         """
#         Check whether all the data are of the same type.
# 
#         Parameters
#         ----------
#         dtype : type
#             if given, check that all data are of this type
#         allowSubclass : bool
#             whether to accept subclasses of the common type
#             default: False
# 
#         Returns
#         -------
#         True if the types are homogeneous, False otherwise
#         """
#         try:
#             for datum in self:
#                 try:
#                     if allowSubclass:
#                         assert isinstance(datum, commontype)
#                     else:
#                         assert type(datum) == commontype
#                 except NameError:
#                     if dtype is None:
#                         commontype = type(datum)
#                     else:
#                         commontype = dtype
#                         if allowSubclass:
#                             assert isinstance(datum, commontype)
#                         else:
#                             assert type(datum) == commontype
#         except AssertionError:
#             return False
#         return True
# 
#     def getHom(self, attr):
#         """
#         Check that the attribute attr is the same for all data and return its
#         value if so.
#         """
#         try:
#             for datum in self:
#                 try:
#                     assert getattr(datum, attr) == attrvalue
#                 except NameError:
#                     attrvalue = getattr(datum, attr)
#         except AssertionError:
#             raise RuntimeError("Attribute '{}' has non-homogeneous values".format(attr))
#         return attrvalue

    ### Processing data in the set ###

    def filter(self, fun):
        """
        Run all data through the given function.

        Parameters
        ----------
        fun : callable of signature datum = fun(datum)

        See also
        --------
        process, map

        Notes
        -----
        This function works in-place. If you need the original data to remain
        unchanged, use process().
        """
        # Have to explicitly run through the data array, because the entries
        # might be reassigned.
        for i, (datum, selected) in enumerate(zip(self._data, self._selected)):
            if selected:
                self._data[i] = fun(datum)

    def process(self, fun):
        """
        Generate a new TaggedSet with filtered data.

        Same as filter(), except that a new list with the processed data is
        returned, while the original one remains unchanged.

        Parameters
        ----------
        fun : callable of signature datum = fun(datum)

        Returns
        -------
        A new list containing the processed data

        See also
        --------
        filter, map

        Notes
        -----
        The new list will contain only the processed data, i.e. data that are
        not in the current selection will not be copied.
        """
        def gen(origin):
            for datum, tags in origin(giveTags=True):
                yield (fun(deepcopy(datum)), deepcopy(tags))

        return TaggedSet(gen(self))

#     def map(self, fun):
#         """
#         Apply fun to all data and return a list of the results.
# 
#         Parameters
#         ----------
#         fun : callable that takes a datum as argument
# 
#         Returns
#         -------
#         A list of fun(datum) for all data in the list
# 
#         Notes
#         -----
#          - If fun does not provide a return value, the output will be a list of
#            None's. Use _ = ... to ignore this output if necessary.
#          - This can be used to manipulate the data in-place; if fun does not
#            manipulate the datum in place but has signature datum = fun(datum),
#            then use apply() instead.
#          - python's built-in map() also works very well with TaggedSets.
#         """
#         return [fun(traj) for traj in self]
