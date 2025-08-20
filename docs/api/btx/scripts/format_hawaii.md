Module btx.scripts.format_hawaii
================================
Some context:

[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles [1]> pwd
/fs/scratch/PAS2136/samuelstevens/datasets/hawaii-beetles
[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles> ls
group_images  images_metadata.csv  individual_specimens  README.md  trait_annotations.csv
[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles> ls group_images/ | head
IMG_0093.png
IMG_0095.png
IMG_0109.png
IMG_0110.png
IMG_0111.png
IMG_0112.png
IMG_0113.png
IMG_0114.png
IMG_0115.png
IMG_0116.png
[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles> ls individual_specimens/ | head
IMG_0093_specimen_1_MECKON_NEON.BET.D20.000001.png
IMG_0093_specimen_2_MECKON_NEON.BET.D20.000003.png
IMG_0093_specimen_3_MECKON_NEON.BET.D20.000004.png
IMG_0095_specimen_1_MECKON_NEON.BET.D20.000005.png
IMG_0095_specimen_2_MECKON_NEON.BET.D20.000007.png
IMG_0095_specimen_3_MECKON_NEON.BET.D20.000010.png
IMG_0109_specimen_1_MECKON_NEON.BET.D20.000011.png
IMG_0109_specimen_2_MECKON_NEON.BET.D20.000017.png
IMG_0109_specimen_3_MECKON_NEON.BET.D20.000026.png
IMG_0110_specimen_1_MECKON_NEON.BET.D20.000035.png
[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles> head trait_annotations.csv
groupImageFilePath,BeetlePosition,individualID,coords_scalebar,coords_elytra_max_length,coords_basal_pronotum_width,coords_elytra_max_width,px_scalebar,px_elytra_max_length,px_basal_pronotum_width,px_elytra_max_width,cm_scalebar,cm_elytra_max_length,cm_basal_pronotum_width,cm_elytra_max_width
group_images/IMG_0093.png,1,NEON.BET.D20.000001,"[[5713.91, 3045.68, 5701.21, 2265.92]]","[[3865.5, 1245.87, 3881.25, 1045.81]]","[[3922.92, 1046.2, 3872.53, 1035.06]]","[[3960.08, 1145.79, 3814.38, 1123.85]]",779.8634159902615,200.67901260470657,51.606702084128464,147.34264012837542,1.0,0.257,0.066,0.189
group_images/IMG_0093.png,2,NEON.BET.D20.000003,"[[5713.91, 3045.68, 5701.21, 2265.92]]","[[3899.77, 2528.67, 3939.62, 2338.98]]","[[3961.39, 2339.49, 3912.19, 2329.79]]","[[3974.62, 2440.72, 3895.98, 2421]]",779.8634159902615,193.8306441200669,50.14708366395775,81.07482963287664,1.0,0.249,0.064,0.104
group_images/IMG_0093.png,3,NEON.BET.D20.000004,"[[5713.91, 3045.68, 5701.21, 2265.92]]","[[3998.51, 3859.69, 4002.25, 3686.53]]","[[4047.58, 3684.09, 4013.58, 3681.39]]","[[4080.85, 3774.4, 3959.48, 3763.86]]",779.8634159902615,173.2003845261319,34.1070373969948,121.82679713429216,1.0,0.222,0.044,0.156
group_images/IMG_0095.png,1,NEON.BET.D20.000005,"[[4361.9, 3001.51, 4356.57, 2218.33]]","[[2727.56, 991.13, 2777.05, 746.39]]","[[2771.23, 737.52, 2719.33, 714.82]]","[[2819.12, 840.3, 2651.68, 832.24]]",783.1981366806234,249.69366772106983,56.64715350306674,167.63387843750445,1.0,0.319,0.072,0.214
group_images/IMG_0095.png,2,NEON.BET.D20.000007,"[[4361.9, 3001.51, 4356.57, 2218.33]]","[[2637.35, 2011.99, 2701.51, 1851.59]]","[[2703.39, 1846.64, 2660.39, 1834.34]]","[[2721.27, 1936.92, 2601.72, 1899.58]]",783.1981366806234,172.7560870128751,44.72460173103842,125.24567098307251,1.0,0.221,0.057,0.16
group_images/IMG_0095.png,3,NEON.BET.D20.000010,"[[4361.9, 3001.51, 4356.57, 2218.33]]","[[2665.12, 2947.47, 2726.98, 2725.72]]","[[2725.91, 2711.99, 2675.41, 2705.29]]","[[2755.34, 2836.81, 2597.89, 2812.56]]",783.1981366806234,230.2166851034043,50.94251662413232,159.30651273566968,1.0,0.294,0.065,0.203
group_images/IMG_0109.png,1,NEON.BET.D20.000011,"[[5963.41, 2293.36, 5928.69, 1510.68]]","[[4369.81, 1151.98, 4434.72, 964.11]]","[[4435.89, 958.48, 4390.69, 941.68]]","[[4453.82, 1063.75, 4319.9, 1024.04]]",783.4497181057634,198.767313711284,48.22115718229985,139.68339378752228,1.0,0.254,0.062,0.178
group_images/IMG_0109.png,2,NEON.BET.D20.000017,"[[5963.41, 2293.36, 5928.69, 1510.68]]","[[4332.92, 1979.73, 4335.18, 1781.63]]","[[4395.78, 1775.08, 4340.7, 1755.66]]","[[4428.95, 1868.03, 4287.33, 1862.43]]",783.4497181057634,198.1128910495225,58.40327730530185,141.73067557871858,1.0,0.253,0.075,0.181
group_images/IMG_0109.png,3,NEON.BET.D20.000026,"[[5963.41, 2293.36, 5928.69, 1510.68]]","[[4264.26, 2738.34, 4313.08, 2537.98]]","[[4362.27, 2538.03, 4321.57, 2526.78]]","[[4386.88, 2641.59, 4329.2, 2638.75, 4324.27, 2637.61, 4245.53, 2614.46]]",783.4497181057634,206.2220211325649,42.2262063178787,139.82246488940592,1.0,0.263,0.054,0.178
[I] samuelstevens@ascend-login01 /f/s/P/s/d/hawaii-beetles> head images_metadata.csv
individualImageFilePath,groupImageFilePath,individualID,taxonID,scientificName,plotID,trapID,plotTrapID,collectDate,ownerInstitutionCode,catalogNumber
individual_specimens/IMG_0093_specimen_1_MECKON_NEON.BET.D20.000001.png,group_images/IMG_0093.png,NEON.BET.D20.000001,MECKON,Mecyclothorax konanus,6,W,006W,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0093_specimen_2_MECKON_NEON.BET.D20.000003.png,group_images/IMG_0093.png,NEON.BET.D20.000003,MECKON,Mecyclothorax konanus,16,W,016W,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0093_specimen_3_MECKON_NEON.BET.D20.000004.png,group_images/IMG_0093.png,NEON.BET.D20.000004,MECKON,Mecyclothorax konanus,6,E,006E,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0095_specimen_1_MECKON_NEON.BET.D20.000005.png,group_images/IMG_0095.png,NEON.BET.D20.000005,MECKON,Mecyclothorax konanus,14,S,014S,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0095_specimen_2_MECKON_NEON.BET.D20.000007.png,group_images/IMG_0095.png,NEON.BET.D20.000007,MECKON,Mecyclothorax konanus,6,E,006E,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0095_specimen_3_MECKON_NEON.BET.D20.000010.png,group_images/IMG_0095.png,NEON.BET.D20.000010,MECKON,Mecyclothorax konanus,14,W,014W,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0109_specimen_1_MECKON_NEON.BET.D20.000011.png,group_images/IMG_0109.png,NEON.BET.D20.000011,MECKON,Mecyclothorax konanus,14,E,014E,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0109_specimen_2_MECKON_NEON.BET.D20.000017.png,group_images/IMG_0109.png,NEON.BET.D20.000017,MECKON,Mecyclothorax konanus,14,S,014S,20190424,NEON,DP1.10022.001
individual_specimens/IMG_0109_specimen_3_MECKON_NEON.BET.D20.000026.png,group_images/IMG_0109.png,NEON.BET.D20.000026,MECKON,Mecyclothorax konanus,14,W,014W,20190424,NEON,DP1.10022.001

Functions
---------

`get_memory_info() ‑> dict[str, float]`
:   Get current memory usage information.

`img_as_arr(img: PIL.Image.Image | pathlib.Path) ‑> jaxtyping.Float[ndarray, 'height width channels']`
:   

`img_as_grayscale(img: PIL.Image.Image | pathlib.Path) ‑> jaxtyping.UInt[ndarray, 'height width']`
:   

`load_img_df(cfg: btx.scripts.format_hawaii.Config) ‑> polars.dataframe.frame.DataFrame`
:   

`load_trait_df(cfg: btx.scripts.format_hawaii.Config) ‑> polars.dataframe.frame.DataFrame`
:   

`main(cfg: btx.scripts.format_hawaii.Config) ‑> int`
:   

`save_example_images(dump_to: pathlib.Path, annotation: btx.scripts.format_hawaii.Annotation, trait_data: dict[str, object]) ‑> None`
:   Save example images with annotations drawn on them.

`worker_fn(cfg: btx.scripts.format_hawaii.Config, group_img_basenames: list[str]) ‑> list[btx.scripts.format_hawaii.Annotation | btx.scripts.format_hawaii.WorkerError]`
:   Worker. Processing group_img_basenames and returns a list of annotations or errors.

Classes
-------

`Annotation(group_img_basename: str, beetle_position: int, group_img_abs_path: pathlib.Path, indiv_img_abs_path: pathlib.Path, indiv_offset_px: tuple[float, float], individual_id: str, ncc: float)`
:   Annotation(group_img_basename: str, beetle_position: int, group_img_abs_path: pathlib.Path, indiv_img_abs_path: pathlib.Path, indiv_offset_px: tuple[float, float], individual_id: str, ncc: float)

    ### Instance variables

    `beetle_position: int`
    :

    `group_img_abs_path: pathlib.Path`
    :

    `group_img_basename: str`
    :

    `indiv_img_abs_path: pathlib.Path`
    :

    `indiv_offset_px: tuple[float, float]`
    :

    `individual_id: str`
    :

    `ncc: float`
    :

    ### Methods

    `to_dict(self) ‑> dict`
    :   Convert annotation to dictionary for JSON serialization.

`Config(hf_root: pathlib.Path = PosixPath('data/hawaii'), log_to: pathlib.Path = PosixPath('logs'), dump_to: pathlib.Path = PosixPath('data/hawaii-formatted'), ignore_errors: bool = False, seed: int = 42, sample_rate: int = 20, slurm_acct: str = '', slurm_partition: str = 'parallel', n_hours: float = 4.0, groups_per_job: int = 4)`
:   Config(hf_root: pathlib.Path = PosixPath('data/hawaii'), log_to: pathlib.Path = PosixPath('logs'), dump_to: pathlib.Path = PosixPath('data/hawaii-formatted'), ignore_errors: bool = False, seed: int = 42, sample_rate: int = 20, slurm_acct: str = '', slurm_partition: str = 'parallel', n_hours: float = 4.0, groups_per_job: int = 4)

    ### Instance variables

    `dump_to: pathlib.Path`
    :   Where to save formatted data.

    `groups_per_job: int`
    :   Number of group images to process per job.

    `hf_root: pathlib.Path`
    :   Where you dumped data when using download_hawaii.py.

    `ignore_errors: bool`
    :   Skip the user error check and always proceed (equivalent to answering 'yes').

    `log_to: pathlib.Path`
    :   Where to save submitit/slurm logs.

    `n_hours: float`
    :   Number of hours to request for each job.

    `sample_rate: int`
    :   Save 1 in sample_rate annotations as example images (default: 1 in 20).

    `seed: int`
    :   Random seed for sampling which annotations to save as examples.

    `slurm_acct: str`
    :   Slurm account to use. If empty, uses DebugExecutor.

    `slurm_partition: str`
    :   Slurm partition to use.

`ImageLoadError(group_img_basename: str, message: str, img_path: str)`
:   Error loading an image file.

    ### Ancestors (in MRO)

    * btx.scripts.format_hawaii.WorkerError
    * builtins.Exception
    * builtins.BaseException

    ### Instance variables

    `img_path: str`
    :

`TemplateMatchingError(group_img_basename: str, message: str, beetle_position: int, indiv_img_path: str)`
:   Error during template matching.

    ### Ancestors (in MRO)

    * btx.scripts.format_hawaii.WorkerError
    * builtins.Exception
    * builtins.BaseException

    ### Instance variables

    `beetle_position: int`
    :

    `indiv_img_path: str`
    :

`WorkerError(group_img_basename: str, message: str)`
:   Base class for worker errors with context.

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException

    ### Descendants

    * btx.scripts.format_hawaii.ImageLoadError
    * btx.scripts.format_hawaii.TemplateMatchingError

    ### Instance variables

    `group_img_basename: str`
    :

    `message: str`
    :