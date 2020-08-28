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

        self.makeSelection()

    def __iter__(self):
        """
        Iterate over all data in current selection. Simply a shortcut for
        self.byTag().
        """
        return self.byTag()

    def __call__(self, *args, **kwargs):
        """
        Synonymous to self.byTag(...)
        """
        return self.byTag(*args, **kwargs)

    def __len__(self):
        """
        Give number of data in current selection
        """
        return sum([self._selection_logic(t in tags for t in self._selection_tags) for tags in self._tags])

    def makeSelection(self, tags='_all', logic=any):
        """
        Mark a subset of the current list as 'active selection'. For most
        purposes, the list will behave as if it contained only these data.

        Input
        -----
        tags : str, list of str, or set of str
            the tags to select
            default: '_all', which is an internal tag attached to all data
        logic : callable (mostly any or all)
            the logic for handling multiple tags. Set this to (the built-in)
            all to select the data being tagged with all the given tags, or to
            any to select the data having any of the given tags
            default: any

        Notes
        -----
        Call this without arguments to reset the selection to the whole
        dataset.
        """
        tags = self.makeTagsSet(tags)
        self._selection_tags = tags
        self._selection_logic = logic

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

    @staticmethod
    def makeTagsList(tags):
        """
        Similar to makeTagsSet(), but keep the order. This is used for plotting
        functions, where the proper alignment of tags and other specifications
        is important. Mostly for internal use.

        Input
        -----
        tags : str, list of str, or set of str

        Output
        ------
        list of str
        """
        if isinstance(tags, str):
            tags = [tags]
        elif isinstance(tags, set):
            tags = list(tags)
        elif not isinstance(tags, list):
            raise ValueError("Did not understand type of 'tags' ({})".format(str(type(tags))))

        return tags

    def getTagsAndLogicFromKwargs(self, yourkwargs):
        """
        Can be used if some function wants to process selection for itself.
        This returns (tags, logic) if they are given, otherwise
        (self._selection_tags, self._selection_logic). If the kwargs are
        present, they are extracted exactly as they are, so you might want to
        run self.makeTagsSet() (or similar) on them. Also, they are really
        extracted from kwargs. Mostly for internal use.
        """
        try:
            logic = yourkwargs['logic']
            del yourkwargs['logic']
        except KeyError:
            if 'tags' in yourkwargs.keys():
                logic = any
            else:
                logic = self._selection_logic
        try:
            tags = yourkwargs['tags']
            del yourkwargs['tags']
        except KeyError:
            tags = self._selection_tags

        return (tags, logic)

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
        """
        tags = self.makeTagsSet(tags)

        if not "_all" in tags:
            tags.add("_all") # A generic tag that addresses all trajectories

        self._data.append(datum)
        self._tags.append(tags)

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
        #tagset = set.union(*[set(), *self._tags]) # This is safe if self._tags is empty
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
        return obj

    def byTag(self, tags=None, logic=None, giveTags=False):
        """
        Returns a generator for all trajectories within the current selection,
        or in the selection specified by the tags and logic arguments. Note
        that in the latter case, the current selection is not overwritten.

        Input
        -----
        tags : str, list of str, or set of str
            the tags we are interested in
        logic : callable
            this should be a function taking an iterable of boolean values and
            returning True or False, i.e. this implements the logic of how to
            deal with multiple tags. The most useful values are the built-ins
            all and any, but customization is possible.
            default: any
        giveTags : boolean
            whether to yield datum or (datum, tags) for each element.
            default: False
        """
        kwargs = {'tags' : tags, 'logic' : logic}
        if tags is None:
            del kwargs['tags']
        if logic is None:
            del kwargs['logic']
        (tags, logic) = self.getTagsAndLogicFromKwargs(kwargs)
        tags = self.makeTagsSet(tags)

        for datum, datumtags in zip(self._data, self._tags):
            if logic(t in datumtags for t in tags):
                if giveTags:
                    yield (datum, datumtags)
                else:
                    yield datum

    def subsetByTag(self, *args, **kwargs):
        """
        A small wrapper for generate(byTag()), i.e. create a subset with
        specific tags. All arguments are forwarded to self.byTag(), so see that
        docstring for more information.
        """
        kwargs['giveTags'] = True
        return type(self).generate(self.byTag(*args, **kwargs))

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
        for i, (datum, tags) in enumerate(zip(self._data, self._tags)):
            if self._selection_logic(t in tags for t in self._selection_tags):
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
