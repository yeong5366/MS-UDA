# MS-UDA

This code is official code of [MS-UDA: Multi-Spectral Unsupervised Domain Adaptation for Thermal Image Semantic Segmentation](https://ieeexplore.ieee.org/document/9468936).

# About Dataset

If you want to use KP dataset, go to [here](https://drive.google.com/file/d/1W6A8IZD2mlP7Olo-Kaoa-kl60PIAN4oI/view?usp=sharing)
And, Please cite our project.

The KP dataset structure is:

    -day2night
    -labels
      -val_day
      -val_night
      -visualize
    -pseudo_KP
      - day
      - night
      - val_day
      - val_night

day2night
> This folder contains fake nighttime thermal images generated by Cycle-GAN, trained with 3283 daytime images and 3095 nighttime images.

labels
> This folder contains real label for KAIST Pedestrian segmentation dataset. The folder 'val_day' contains the daytime 503 rgb-t images which is used to train the supervised network for comparison. The folder 'val_night' contains the nighttime 447 rgb-t images which is used to validate MS-UDA. The folder 'visualize' contains visualization of the label

pseudo_KP
> This folder contains the rgb-t input images and their pseudo-labels. The folder 'day' contains the 3283 daytime rgb-t images for domain adaptation and day-to-night translation. The folder 'night' contains the 3095 nighttime rgb-t images for day-to-night translation. The folder 'val_day' contains the 503 rgb-t daytime images. The folder 'val_night' contains the 447 rgb-t nighttime images. 

This [folder](https://github.com/yeong5366/MS-UDA/tree/main/filenames_KP) contains the filename for KP dataset.   
It contains:

    -day_rgb/th.txt
    -night_rgb/th.txt
    -val_day_rgb/th.txt
    -val_night_rgb/th.txt

day_rgb/th.txt has the name of 3283 daytime rgb/thermal images for UDA and day-to-night translation.   
night_rgb/th.txt has the name of 3095 nighttime rgb/thermal images which is used for day-night translation.  
val_day_rgb/th.txt has the name of 503 daytime rgb/thermal images for supervised learning method to compare the results with UDA.  
val_night_rgb/th.txt has the name of 447 nighttime rgb/thermal images for validation.  

'..pred.png' is the visualized pseudo-labels. '..pseudo.png' are pseudo labels. '...rgb/th.png' are the input images.
