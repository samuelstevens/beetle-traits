Module btx.modeling.frozen
==========================

Classes
-------

`Frozen(dinov3_ckpt: pathlib.Path = PosixPath('models/dinov3_vitb16.eqx'))`
:   Frozen(dinov3_ckpt: pathlib.Path = PosixPath('models/dinov3_vitb16.eqx'))

    ### Instance variables

    `dinov3_ckpt: pathlib.Path`
    :

`Model(cfg: btx.modeling.frozen.Frozen, *, key: jax.Array)`
:   Model(cfg: btx.modeling.frozen.Frozen, *, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `head: equinox.nn._mlp.MLP`
    :

    `vit: btx.modeling.dinov3.VisionTransformer`
    :