# put origin images into ./jpg/
python3 alignment_MTB.py -d jpg/
python3 hdr_and_tone_mapping.py -i 1 -d jpg/
python3 hdr_and_tone_mapping.py -i 2 -d jpg/
python3 hdr_and_tone_mapping.py -i 3 -d jpg/
