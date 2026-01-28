import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import equinox as eqx
    import jax
    import jax.nn as jnn
    import jax.numpy as jnp
    import marimo as mo
    import optax

    return eqx, jax, jnn, jnp, mo, optax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Freezing Parameters in Equinox
    There are 3 main steps:
    1. Create a boolean pytree which is a mirror of your model, and specify which parameters should not update by setting their leaves to false
    2. Partion your model into the trainable vs frozen model
    3. Calculate and update the gradients of the trainable model
    """)
    return


@app.cell
def _(eqx, jax, jnn, mo):
    class Model(eqx.Module):
        backbone: eqx.nn.MLP
        head: eqx.nn.Linear

        def __init__(self, in_size, hidden_size, out_size, key):
            k1, k2 = jax.random.split(key)
            self.backbone = eqx.nn.MLP(
                in_size=in_size,
                out_size=hidden_size,
                width_size=4,
                depth=2,
                activation=jnn.gelu,
                key=k1,
            )
            self.head = eqx.nn.Linear(
                in_features=hidden_size, out_features=out_size, key=k2
            )

        def __call__(self, x):
            features = self.backbone(x)
            return self.head(features)

    key = jax.random.PRNGKey(0)
    model = Model(2, 4, 2, key)
    mo.md(
        "**first set up the model:** in this case the backbone is a MLP with a linear head"
    )
    return (model,)


@app.cell
def _(eqx, jax, mo, model):
    # Step 1 create the filter_specification
    filter_spec = jax.tree_util.tree_map(
        lambda _: False, model
    )  # creates a pytree with the same shape as model setting each leaf to false
    filter_spec = eqx.tree_at(
        where=lambda tree: (
            tree.head
        ),  # if a leaf is part of the head, make it trainable by setting it equal to true
        pytree=filter_spec,
        replace=jax.tree_util.tree_map(eqx.is_array, model.head),
    )
    mo.md(
        "**Step 1:** Creates a pytree with the same shape as the model, with each leaf set to false. Set only the trainable leaves to true. In this case, we set the leaves from the head to true."
    )
    return (filter_spec,)


@app.cell
def _(jax, mo):
    # Now to generate data
    def get_data(dataset_size, key):
        k_x, k_y = jax.random.split(key)
        x = jax.random.normal(k_x, (dataset_size, 2))  # input data
        y = jax.random.normal(k_y, (dataset_size, 2))  # output data
        return x, y

    # Generate a small batch
    data_key = jax.random.PRNGKey(42)
    x_data, y_data = get_data(32, data_key)

    mo.md(f"Generated data shape: x={x_data.shape}, y={y_data.shape}")
    return x_data, y_data


@app.cell
def _(eqx, jax, jnp, mo):
    # Standard mse
    def loss_fn(diff_model, static_model, x, y):
        model = eqx.combine(diff_model, static_model)
        preds = jax.vmap(model)(x)
        return jnp.mean((preds - y) ** 2)

    @eqx.filter_jit
    def step_model(model, x, y, optim, state, filter_spec):
        # Step 2, partition the model using the filters_spec
        diff_model, static_model = eqx.partition(model, filter_spec)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_model, static_model, x, y)

        # # Step 3, only update the gradients on the trainable model, and recombine
        updates, state = optim.update(grads, state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)

        # recombine the models
        model = eqx.combine(diff_model, static_model)
        return model, state, loss

    mo.md("""
    **Step 2: Partition the model** \n
    The leaves which are true based on the filter_spec are included in diff_model (head) and the leaves which are false are in the static_model

    **Step 3: Update trainable model, and recombine** \n
    Since the models are separated, only the non-frozen model can be updated. After doing this, the models are recombined.
    """)
    return (step_model,)


@app.cell
def _(eqx, filter_spec, mo, model, optax, step_model, x_data, y_data):
    # now set up the model and go through one step
    optim = optax.adam(learning_rate=0.01)

    # use only the differentiatable model to set up the optimizer state
    diff_model, static_model = eqx.partition(model, filter_spec)
    opt_state = optim.init(diff_model)

    old_backbone_weights = model.backbone.layers[0].weight
    old_head_weights = model.head.weight

    new_model, opt_state, loss = step_model(
        model, x_data, y_data, optim, opt_state, filter_spec
    )

    new_backbone_weights = new_model.backbone.layers[0].weight
    new_head_weights = new_model.head.weight

    mo.md(f"""
    #Results
    ###1st Layer of the frozen backbone
    **before step:** {old_backbone_weights} \n
    **after step:** {new_backbone_weights}
    ### Trainable head
    **before step:** {old_head_weights} \n
    **after step:** {new_head_weights}

    """)
    return


if __name__ == "__main__":
    app.run()
