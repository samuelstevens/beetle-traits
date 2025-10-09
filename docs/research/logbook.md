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

# 08/29/2025

- [done] Point MAE px
- [done] Point MAE cm
- [done] Line MAE px
- [done] Line MAE cm
- [done] Mean line absolute error cm over validation
- [done] Median line absolute error MAE cm over validation
- [done] Maximum line absolute error MAE cm over validation

Actually, my DINOv3 model is not frozen.
So I probably should fix that.
Regardless, I will let these two jobs finish, then continue to improve the DINOv3-based model.

# 08/31/2025

Both my runs failed to improve at all.

Claude suggests:

- [done] utils.Resize swaps width and height.
- No momentum on SGD. [fixed by using adamw]
- [done] frozen.Model has a mistake in the einops.rearrange expression

GPT-5 suggests:

- [done] utils.Resize swaps width and height.
- [done] toy.Model has a mistake in the einops.rearrange expression
- [done] frozen.Model is not really frozen
- [done] dinov3.PatchEmbed has a typo in the kwargs to einops.rearrange
- dinov3.LayerScale.gamma should not be initialized to 0s for training from scratch.
- train.py: Dataset builds points_px_l22 as [width, length], but in plot_preds you unpack gt_length_px, gt_width_px = batch["points_px"][0], i.e., reversed. This only affects visualization (not the loss/MSE)

Now I have some other problems to work through.

1. Datasets. I need integrate the BeetlePalooza data first.
2. Objective. I need to add the gaussian heatmap regression.
3. We also need a better decoder using the upsampling approach.
4. Then better models (EoMT) if it's still not good enough.

[Some suggestions for the keypoint problem](https://chatgpt.com/share/68b5d006-d3dc-8003-ae63-dc8690c0a97f)

# 09/04/2025

The TODOs have not changed from above.
I could ask the team to work on data annotation of the BioRepo data.
But I should try to figure out which samples are most meaningful.
Maybe stratify by species and pick out 1 of each species?

Meeting Notes:

- Metrics: validation RMSE, MAE, bias (consistently under or over).
- Stratify by genus instead of species.
- Annotate two trays of non-hawaii-looking beetles for validation.

# 09/23/2025

- Validation is only on the Hawaii beetles
- Need to use the BeetlePalooza data.

The BeetlePalooza data is shit, unless you use the images resized from Isa. But if the images are resized, then I won't be able to do pixel-level matching. So I need to write a clear bug report.

So here is my bug report:


Goal: I want individual beetle pictures with pixel-level annotations.

Attempts:

1. Map the annotations from `BeetleMeasurements.csv` to the group images. These don't match. See attached `A00000032929.png`. This is from the main branch, commit [b1dd26a9fbb058ad6b8ce62731c9ccbeba320485](https://huggingface.co/datasets/imageomics/2018-NEON-beetles/tree/b1dd26a9fbb058ad6b8ce62731c9ccbeba320485).
2. Use the group images in `/fs/ess/PAS2136/BeetlePalooza-2024/Resized Images [Corrected from ISA]` on OSC. These match the annotations. However, I need to know the individual beetle image positions in the group image so I can offset the annotations from the group image to the individual images. In the past, I've used [`skimage.feature.match_template()`](https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.match_template) to do this. But I don't get a good match between the individual images and the resized group images. In contrast, I get a perfect match between the individual images and the original group images.

Things I have yet to try:

1. Resize the individual images using the same resizing ratio from the original group image to the resized group image. Downside: this requires both the original group image that produced the indvidual image crops AND the resized group image that matches the annotations to calculate the resizing ratio.
2. Use the resized images and simply add padding to the annotations to re-crop individual images from the group images. Downside: this is a heuristic and throws away all of the work done cropping the indvidiuals originally.
3. Downloading pr/26 and praying that one of my original attempts works.

Do you have any recommendations?


# 10/09/2025

Supposedly pr/31 fixes the issue above.
Now I need to onboard Hanane.

While I work on data, she can work on:

1. (me) data
2. Objective. I need to add the gaussian heatmap regression.
3. We also need a better decoder using the upsampling approach.
4. Then better models (EoMT) if it's still not good enough.
