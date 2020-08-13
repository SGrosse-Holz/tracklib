class TaggedList:
    """
    A list with a series of tags for each object. The idea is to use this as a
    many-to-many dict.

    This class also works as an iterator, in itself (running through all
    trajectories) or by using byTag() to loop over subsets.

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

    def __iter__(self, giveTags=False):
        """
        Iterate over all data. Note that this is really just a short cut,
        equivalent to byTag('_all'). Therefore there is no option for yielding
        the tags as well.
        """
        return self.byTag('_all')

    def __len__(self):
        return len(self._data)

    @staticmethod
    def makeTagsSet(tags):
        """
        An input processing function making sure that the frequently used
        'tags' argument is a set of strings. Mostly for internal use.
        """
        if isinstance(tags, str):
            tags = set([tags])
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
        tags) pairs.
        """
        obj = cls()
        for (datum, tags) in iterator:
            obj.append(datum, tags)
        return obj

    def byTag(self, tags, logic=all, giveTags=False):
        """
        Returns a generator for all trajectories having certain tags. The
        behavior for multiple tags can be controlled with the logic argument,
        whose two most useful values are the built-in functions all or any. The
        default is logic=all, i.e. we are interested in the data tagged with
        all the tags given.

        Input
        -----
        tags : list/set of tags or single tag
            the tags we are interested in
        logic : callable
            this should be a function taking an iterable of boolean values and
            returning True or False, i.e. this implements the logic of how to
            deal with multiple tags. The most useful values are the built-ins
            all and any, but customization is possible.
            default: all
        giveTags : boolean
            whether to yield datum or (datum, tags) for each element.
            default: False
        """
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
