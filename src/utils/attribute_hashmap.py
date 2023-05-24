class AttributeHashmap(dict):
    """
    A Specialized hashmap such that:
        hash_map = AttributeHashmap(dictionary)
        `hash_map.key` is equivalent to `dictionary[key]`

    Credit to https://stackoverflow.com/a/14620633
    """
    def __init__(self, *args, **kwargs):
        super(AttributeHashmap, self).__init__(*args, **kwargs)
        self.__dict__ = self
