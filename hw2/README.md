## VFX - hw 2
#### Image Stiching

Implemnt image stiching majorly based on [1] to generate panoramas. The process is :

(0) Put images belonging to the same panorama in a directory which contains `focal.txt` that describes the focal length of each image.

(1) Do cylindrical warping according to `focal.txt` and output warpped images into `<data_dir>/warp`.

(2) Do feature detection using MSOP [2] and match the feature points for every two consecutive images. Output matching_images and `feature_matching.npy` under `<data_dir>/warp`.

(3) Align consecutive images according to `feature_matching.npy` by RANSAC and assume there is only translation between images. Output `pairwise_alignment.npy` under `<data_dir>/warp`.

(4) Stich images and blend the overlapped areas to draw a panorama. Also do global warping to alleviate the drift problem and crop the margin to refine the result panorama.

###  Requirements:
- macOS or Linux
- python 3.4+
- OpenCV 3.2+
- numpy 1.13+
- scipy 0.19+
- tqdm 4.21+

### Execution:
Put the images and `focal.txt` under `<data_dir>`; the naming of image file should follow `<IMG_NUM>.jpg`. Then, `./run.sh <data_dir>`. Sample images are provided and detailed information of program execution can be found in the head of each file.

### Reference:
[1] M. Brown, D. G. Lowe, ’Recognising Panoramas’, ICCV 2003.

[2] Matthew Brown, Richard Szeliski, Simon Winder, ’Multi-Image Matching using Multi-Scale Oriented Patches’, CVPR 2005.
