Module btx.modeling.frozen
==========================

Classes
-------

`Frozen(img_size: int = 256, dinov3_ckpt: pathlib.Path = PosixPath('models/dinov3_vitb16.eqx'))`
:   Frozen(img_size: int = 256, dinov3_ckpt: pathlib.Path = PosixPath('models/dinov3_vitb16.eqx'))

    ### Instance variables

    `dinov3_ckpt: pathlib.Path`
    :

    `img_size: int`
    :

`Model(cfg: btx.modeling.frozen.Frozen, *, key: jax.Array)`
:   Model(cfg: btx.modeling.frozen.Frozen, *, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `head: equinox.nn._linear.Linear`
    :

    `vit: btx.modeling.dinov3.VisionTransformer`
    :