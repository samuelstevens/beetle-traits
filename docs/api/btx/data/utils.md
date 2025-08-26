Module btx.data.utils
=====================

Classes
-------

`DecodeRGB()`
:   DecodeRGB()

    ### Ancestors (in MRO)

    * grain._src.core.transforms.MapTransform
    * abc.ABC

    ### Methods

    `map(self, sample: btx.data.utils.Sample) ‑> btx.data.utils.Sample`
    :

`Resize(size: int = 256)`
:   Resize(size: int = 256)

    ### Ancestors (in MRO)

    * grain._src.core.transforms.MapTransform
    * abc.ABC

    ### Instance variables

    `size: int`
    :

    ### Methods

    `map(self, sample: dict[str, object]) ‑> dict[str, object]`
    :   Maps a single element.

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

    `points_px: jaxtyping.Float[ndarray, '2 2 2']`
    :   {width, length} x two points x {x, y}.