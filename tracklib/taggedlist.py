from copy import deepcopy

class TaggedList:
    """
    A list with a set of tags for each object. The idea is to use this as a
    many-to-many dict.

    This class can be used as an iterator in constructs such as
        for datum in list: ...
    The behavior of this iterator can be adjusted in multiple ways: by
    previously selecting only a subset of the list using makeSelection() or by
    explicitly giving these selection arguments to byTag() or (equivalently)
    the call syntax (i.e. for datum in list(...): ...).

    Because it is the most explicit, the actual functionality is implemented in
    byTag(), while the other methods are just aliases for this.

    Notes
    -----
    We chose to not implement the full sequence interface for multiple reasons:
     - this should be thought of more like a dictionary, i.e. the sequential
       nature of the data is (presumably) not important, except for matching
       (datum, tags) pairs
     - not implementing the interface allows us to overload the item access
       operator [] in a more useful way (TODO)
    """
    def __init__(self):
        """
        Create a new, empty list
        """
        self._data = []
        self._tags = []
        self._selected = []

    def __iter__(self):
        """
        Iterate over all data in current selection. Simply a shortcut for
        self(), i.e. the call syntax.
        """
        return self()

    def __call__(self, giveTags=False):
        """
        Iterate through the current selection of the list.

        Input
        -----
        giveTags : bool
            whether to return the tags or only the data
            default: False

        Output
        ------
        A generator, yielding either just the list entries, or (data, tags)
        pairs, depending on giveTags
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

    def makeSelection(self, **kwargs):
        """
        Mark a subset of the current list as 'active selection'. For most
        purposes, the list will behave as if it contained only these data.

        There are multiple ways to select data. Which one is used depends on
        the kwargs given. With increasing preference, these methods are
         - select by tag: use the kwargs 'tags' and 'logic'
         - select with a user-specified function: use kwarg 'selector'

        Without any additional input, makeSelection() will select the whole
        list. This can be used to reset selection.

        Input
        -----
        Here we give more detailed descriptions of the possible kwargs. For
        when to use which, see above.
        tags : str, list of str, or set of str
            the tags to select. How these go together will be determined by
            'logic'.
        logic : callable (most useful: the built-ins any() and all())
            the logic for handling multiple tags. Set this to (the built-in)
            all to select the data being tagged with all the given tags, or to
            any to select the data having any of the given tags.
            default: any
        selector : callable
            should expect the datum and a set of tags as input and return True
            if the datum is to be selected, False otherwise.

        Further kwargs
        refining : bool
            set to True to apply the current selection scheme only to those
            data that are already part of the selection. This can be used to
            successively refine a selection by using different selection
            methods (e.g. first by tag, then by some other criterion)
            default: False

        Notes
        -----
        Call this without arguments to reset the selection to the whole
        dataset.
        """
        assert len(self._data) == len(self._tags) == len(self._selected)
        if not 'refining' in kwargs.keys():
            kwargs['refining'] = False

        # Define selector function according to given arguments
        if 'selector' in kwargs.keys():
            selector = kwargs['selector']
        elif 'tags' in kwargs.keys():
            kwargs['tags'] = TaggedList.makeTagsSet(kwargs['tags'])
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

    @staticmethod
    def makeTagsSet(tags):
        """
        An input processing function making sure that the frequently used
        'tags' argument is a set of strings. Mostly for internal use.

        Input
        -----
        tags : str, list of str, or set of str

        Output
        ------
        set of str
        """
        if isinstance(tags, str):
            tags = {tags}
        elif isinstance(tags, list):
            tags = set(tags)
        elif not isinstance(tags, set):
            raise ValueError("Did not understand type of 'tags' ({})".format(str(type(tags))))

        return tags

    def append(self, datum, tags=set()):
        """
        Append the given datum with the given tags.

        Input
        -----
        datum : the datum to append
        tags : str, list of str, or set of str
            the tags to attach to this new datum

        Notes
        -----
        The generic tag '_all' will be added to all data.
        The newly added datum will not be part of the current selection.
        """
        tags = self.makeTagsSet(tags)

        if not "_all" in tags:
            tags.add("_all") # A generic tag that addresses all trajectories

        self._data.append(datum)
        self._tags.append(tags)
        self._selected.append(False)

    def mergein(self, other, additionalTags=set()):
        """
        Add the contents of the TaggedList 'other' to the caller.

        Input
        -----
        other : TaggedList
            the list whose data to add
        additionalTags : str, list of str, or set of str
            additional tag(s) to add to all of the new data.
        """
        if not issubclass(type(other), TaggedList):
            raise TypeError("Can only merge a TaggedList")

        additionalTags = TaggedList.makeTagsSet(additionalTags)
        newTags = [tags | additionalTags for tags in other._tags]

        self._data += other._data
        self._tags += newTags
        self._selected += other._selected

    def addTags(self, tags):
        """
        Add new tag(s) to all data in the current selection

        Input
        -----
        tags : str, list of str, or set of str
            the tag(s) to add
        """
        tags = TaggedList.makeTagsSet(tags)

        for _, curtags in self(giveTags=True):
            curtags |= tags

    def tagset(self, omit_all=True):
        """
        Return the set of all tags in the current selection.

        Input
        -----
        omit_all : bool
            whether to omit the generic '_all' tag from the output
            default: True

        Output
        ------
        Set of all tags in the current selection
        """
        tagset = set()
        for _, tags in self(giveTags=True):
            tagset |= tags

        if omit_all:
            return tagset - {"_all"}
        else:
            return tagset

    @classmethod
    def generate(cls, iterator):
        """
        Generate a new TaggedList from an iterator/generator yielding (datum,
        tags) pairs. If you do not want to attach specific tags, set tags=[].
        """
        obj = cls()
        for (datum, tags) in iterator:
            obj.append(datum, tags)

        obj.makeSelection()
        return obj

    def isHomogeneous(self, dtype=None, allowSubclass=False):
        """
        Check whether all the data are of the same type.

        Input
        -----
        dtype : type
            if given, check that all data are of this type
        allowSubclass : bool
            whether to accept subclasses of the common type
            default: False

        Output
        ------
        True if the types are homogeneous, False otherwise
        """
        try:
            for datum in self:
                try:
                    if allowSubclass:
                        assert isinstance(datum, commontype)
                    else:
                        assert type(datum) == commontype
                except NameError:
                    if dtype is None:
                        commontype = type(datum)
                    else:
                        commontype = dtype
                        if allowSubclass:
                            assert isinstance(datum, commontype)
                        else:
                            assert type(datum) == commontype
        except AssertionError:
            return False
        return True

    def getHom(self, attr):
        """
        Check that the attribute attr is the same for all data and return its
        value if so.
        """
        try:
            for datum in self:
                try:
                    assert getattr(datum, attr) == attrvalue
                except NameError:
                    attrvalue = getattr(datum, attr)
        except AssertionError:
            raise RuntimeError("Attribute '{}' has non-homogeneous values".format(attr))
        return attrvalue

    def apply(self, fun):
        """
        Apply fun to all data.

        Input
        -----
        fun : callable of signature datum = fun(datum)

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
        Same as apply(), except that a new list with the processed data is
        returned, while the original one remains unchanged.

        Input
        -----
        fun : callable of signature datum = fun(datum)

        Output
        ------
        A new list containing the processed data

        Notes
        -----
        The new list will contain only the processed data, i.e. data that are
        not in the current selection will not be copied.
        """
        def gen(origin):
            for datum, tags in origin(giveTags=True):
                newdat = deepcopy(datum)
                yield (fun(newdat), tags)

        return TaggedList.generate(gen(self))
