Module btx.modeling.toy
=======================

Classes
-------

`Model(cfg: btx.modeling.toy.Toy, *, key: jax.Array)`
:   Model(cfg: btx.modeling.toy.Toy, *, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `head: equinox.nn._linear.Linear`
    :

    `patch: btx.modeling.toy.PatchEmbed`
    :

`PatchEmbed(patch: int = 16, d_model: int = 192, img_size: int = 256, *, key: jax.Array)`
:   PatchEmbed(patch: int = 16, d_model: int = 192, img_size: int = 256, *, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `pos: jaxtyping.Float[Array, 'n_patches+1 d_model']`
    :

    `proj: equinox.nn._conv.Conv2d`
    :

`Toy(img_size: int = 256, patch: int = 16, d_model: int = 192)`
:   Toy(img_size: int = 256, patch: int = 16, d_model: int = 192)

    ### Instance variables

    `d_model: int`
    :

    `img_size: int`
    :

    `patch: int`
    :