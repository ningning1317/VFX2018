# Usage: ./run.sh <data_dir>
# put origin images into <data_dir>
#
# put `focal.txt` in <data_dir>, which records the focal lengths of the images
#
# default setting of the script will generate
# the panorama in <data_dir>/warp/refine_panorama.png
#
# if the images are not provided in correct order (by left to right),
# add --random_order T when execute harris.py

python3 cylindrical_warping.py -d "$1"
python3 harris.py -d "$1/warp"
python3 pairwise_alignment.py -d "$1/warp"
python3 align_and_blend.py -d "$1/warp" --refine
