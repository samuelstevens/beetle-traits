Module btx.data.utils
=====================

Classes
-------

`Config()`
:   Helper class that provides a standard way to create an ABC using
    inheritance.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * btx.data.beetlepalooza.Config
    * btx.data.biorepo.Config
    * btx.data.hawaii.Config

    ### Instance variables

    `dataset: type['Dataset']`
    :

    `key: str`
    :

`Dataset(*args, **kwargs)`
:   Interface for datasources where storage supports efficient random access.
    
    Note that `__repr__` has to be additionally implemented to make checkpointing
    work with this source.

    ### Ancestors (in MRO)

    * grain._src.python.data_sources.RandomAccessDataSource
    * typing.Protocol
    * typing.Generic
    * abc.ABC

    ### Descendants

    * btx.data.beetlepalooza.Dataset
    * btx.data.biorepo.Dataset
    * btx.data.hawaii.Dataset

    ### Instance variables

    `cfg: btx.data.utils.Config`
    :

`Sample(*args, **kwargs)`
:   dict() -> new empty dictionary
    dict(mapping) -> new dictionary initialized from a mapping object's
        (key, value) pairs
    dict(iterable) -> new dictionary initialized as if via:
        d = {}
        for k, v in iterable:
            d[k] = v
    dict(**kwargs) -> new dictionary initialized with the name=value pairs
        in the keyword argument list.  For example:  dict(one=1, two=2)

    ### Ancestors (in MRO)

    * builtins.dict

    ### Class variables

    `beetle_id: str`
    :

    `beetle_position: int`
    :

    `group_img_basename: str`
    :

    `img_fpath: str`
    :

    `loss_mask: jaxtyping.Float[ndarray, '2']`
    :   Mask for {width, length} indicating which measurements to train on. 1.0 = train, 0.0 = skip.

    `points_px: jaxtyping.Float[ndarray, 'lines 2 2']`
    :   {width, length} x two points x {x, y}.

    `scalebar_px: jaxtyping.Float[ndarray, '2 2']`
    :   two points x {x, y}.

    `scalebar_valid: jaxtyping.Bool[ndarray, '']`
    :   Whether scalebar is usable for converting pixel errors to cm.

    `scientific_name: str`
    :

    `split: str`
    :