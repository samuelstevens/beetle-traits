# 06/27/2025

I am setting this repo up so that others can contribute to this development effort.

What do I need to know in order to get this figured out?

1. Packaging status
2. Code layout
3. At least one specific problem that needs to be solved.

# 07/18/2025

As I understand it, we have some labeled data with elytra width and lengths from both Hawaii/PUUM and BeetlePalooza.
We can use these examples to make a training/validation split, and then we will apply the trained model to images from Arizona and we will get a bunch of trait data.
This is a fairly straightforward machine learning problem, with the specific challenge of distribution shift.
We have an expected maximum accepted error/variance in the predictions due to variance in human annotations.
That is, we don't need the machine to be *more* precise than human annotations.

1. Train a (pre-trained?) model on a training split of the annotated data from Hawaii and BeetlePalooza.
2. Evaluate on the validation split of the annotated data from Hawaii and BeetlePalooza.
3. Make predictions on the test split of (as yet unannotated) data from Arizona.
4. Evaluate the prediction quality.
5. Repeat this loop until the prediction quality is good enough.

How can we improve prediction quality?

- Annotating more data, especially in-domain data.
- Using better or larger pre-trained models.
- Using better training algorithms, hyperparameters, etc.

# 07/23/2025

Can we train a small model, given the images and keypoints, to predict the keypoints on unseen images?

How am I simplifying the task?

- Hawaii only
- No pre-trained model
- No complex architecture
- No species generalization

What does this enable me to focus on?

- Training loop
- Learning Jax's new NamedShard
- Visualizing predictions (in wandb)
- Writing a good dataloader for Hawaii
