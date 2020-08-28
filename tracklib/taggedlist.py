from copy import deepcopy

class TaggedList:
    """
    A list with a series of tags for each object. The idea is to use this as a
    many-to-many dict.

    This class also works as an iterator, in itself (running through all
    trajectories) or by using byTag() to loop over subsets.

    The behavior of __iter__() can be adjusted with makeSelection() : this
    allows for pre-selecting a certain set of tags, such that methods that
    subsequently use constructs like `for datum in list:` only work on the
    selected data. This is especially useful when subclassing this class,
    because we don't have to re-implement the selection logic.
    Note that if we need the tags associated with the data as well, we still
    have to explicitly call byTag(), but we can use the selection values.

    Notes
    -----
    We chose to not implement the full sequence interface for multiple reasons:
     - this should be thought of more like a dictionary, i.e. the sequential
       nature of the data is (presumably) not important, except for matching
       (datum, tags) pairs
     - not implementing the interface allows us to overload the item access
       operator [] in a more useful way
    """
    def __init__(self):
        self._data = []
        self._tags = []

        self.makeSelection()

    def __iter__(self):
        """
        Iterate over all data. Note that this is really just a short cut,
        equivalent to byTag(<selection>). Therefore there is no option for
        yielding the tags as well.
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
        # return len(self._data)

    def makeSelection(self, tags='_all', logic=any):
        tags = self.makeTagsSet(tags)
        self._selection_tags = tags
        self._selection_logic = logic

    @staticmethod
    def makeTagsSet(tags):
        """
        An input processing function making sure that the frequently used
        'tags' argument is a set of strings. Mostly for internal use.
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
        is important.
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
        extracted from kwargs.
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

    def append(self, datum, tags=[]):
        """
        Append the given datum with the given tags. tags can be a single
        string, a list, a set, or omitted (i.e. empty). The generic tag "_all"
        will be attached to all data.
        """
        tags = self.makeTagsSet(tags)

        if not "_all" in tags:
            tags.add("_all") # A generic tag that addresses all trajectories

        self._data.append(datum)
        self._tags.append(tags)

    def mergein(self, other, additionalTags=set()):
        """
        Add the contents of the TaggedList 'other' to the caller.

        With the 'additionalTags' argument, we have the option to provide some
        additional tags that will be attached to all the newly added data.
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
        Return a set of all available tags. By default, we omit the generic
        "_all" tag, i.e. return only the user-assigned tags. Set omit_all=False
        to return all tags.
        """
        tagset = set.union(*[set(), *self._tags]) # This is safe if self._tags is empty
        if omit_all:
            return tagset - set(["_all"])
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
        Returns a generator for all trajectories having certain tags. The
        behavior for multiple tags can be controlled with the logic argument,
        whose two most useful values are the built-in functions all or any. The
        default is logic=all, i.e. we are interested in the data tagged with
        all the tags given.

        If called without specifying tags, this will respect the selection made
        with makeSelection()

        Input
        -----
        tags : list/set of tags or single tag
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
        Check whether the data type of all the data is the same and equal to
        dtype if given. To also allow subclasses of dtype, set
        allowSubclass=True.
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

        Note: this returns a new list of only the data within the current
        selection (after processing).
        """
        def gen(origin):
            for datum, tags in origin(giveTags=True):
                newdat = deepcopy(datum)
                yield (fun(newdat), tags)

        return TaggedList.generate(gen(self))
