## VFX - hw 1
#### High Dynamic Range Imaging

Implemnt MTB [1] for image alignment, HDR by [2] and Photographic Tone mapping by [3]. Given a image series with different exposure, our program outputs aligned images, Response Curve Plot, a .hdr image and a tonemapped LDR image.

###  Requirements:
- python 3.4+
- OpenCV 3.2+
- numpy 1.13+
- matplotlib 2.0+
- tqdm 4.21+

### Execution:
put the images under `./jpg`, the naming of image file should follow `<IMG_NUM>_<shutter>.jpg`. Then, `./run.sh`. Sample images are provided and detailed information of program execution can be found in the head of `alignment_MTB.py` and `hdr_and_tone_mapping.py`.

### Reference:
[1] Greg Ward, ‘Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Handheld Exposures’, Journal of Graphics Tools, vol. 8, pp. 17-30, 2003.

[2] Paul E. Debevec, Jitendra Malik, ‘Recovering High Dynamic Range Radiance Maps from Photographs’, SIGGRAPH, 1997.

[3] Reinhard, Erik, et al. ‘Photographic tone reproduction for digital images’, ACM transactions on graphics (TOG) 21.3 (2002): 267-276.
