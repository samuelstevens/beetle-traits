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

Update: downloaded Hawaii data. Individual annotations are missing, and some group trait annotations are wrong. Asked both Mridul and Rayeed about both of these issues. See `notebooks/hawaii.py` for code.

I plan on using `grain` and following this tutorial: https://google-grain.readthedocs.io/en/latest/tutorials/dataset_basic_tutorial.html

# 08/12/2025

Okay. I used match_template from skimage.feature to get individual annotations.
All this information is calculated by:

uv run src/btx/scripts/format_hawaii.py --ignore-errors --sample-rate 5 --hf-root /fs/scratch/PAS2136/samuelstevens/datasets/hawaii-beetles/ --slurm-acct PAS2136 --slurm-partition nextgen --n-hours 4

And is saved to `data/hawaii-formatted/annotations.json`.

Now we need to build a dataloader for this data.

# 08/18/2025

The dataloader is mostly done.
Now I have to do a couple things to start getting real results.

1. train/val split so that we can meaningfully track progress.
2. Run it on a slurm node rather than the login node.

I actually think that even with just using all the data, I would like to see that we can meaningfully overfit just one batch. -> we can! Using just a patch embed + a linear layer is enough with a batch size of 2 and SGD.

So I need to read the Claude and ChatGPT research reports.
Then we can use DINOv3 + an MLP + whatever loss term the LLMs think is correct.

# 08/20/2025

Now that I've passed candidacy, I can work on this again!
So, we have EoMT, Gaussian heatmap regression, DINOv3, train/val splits.
What to work on first?

1. Train/val split. [done]
2. Gaussian heatmap regression
3. DINOv3
4. EoMT


# 08/22/2025

- submitit+slurm

I think that resizing makes the most sense.
The metrics that I want to track:

- {line, point} error in {px, cm}

# 08/25/2025

What can I do on beetles that's easy?
Let's adapt DINOv3 to Jax.

# 08/26/2025

I have a Jax version of DINOv3.
Now I need to feed its patch-level features as input to my dumb little MLP for predicing (x, y).
This seems to be working quite well.

# 08/28/2025

I need to measure line and point error in px and cm during training.
I also need to measure mean, median and maximum error in cm over the entire validation set.
Once I do this, then I can train a toy model on Slurm.
Then I can train a DINOV3 model on Slurm.
Each of these can run for 6 hours.
Then we can discuss ways to make this system much stronger.
This includes things like additional data (BeetlePalooza), better objectives (Gaussian heatmap regression), better decoders (upsampling?), better models (EoMT) and better hyperparameters.
We also need to annotate some of the BioRepo data as a test set.
