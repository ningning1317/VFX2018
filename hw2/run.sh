# put origin images into ./imgs
#
# put `focal.txt` in ./imgs, which records the focal lengths of the images
#
# default setting of the script will generate
# the panorama in ./imgs/warp/refine_panorama.png

python3 cylindrical_warping.py -d ./imgs/
python3 harris.py -d ./imgs/warp
python3 pairwise_alignment.py -d ./imgs/warp
python3 align_and_blend.py -d ./imgs/warp --refine
