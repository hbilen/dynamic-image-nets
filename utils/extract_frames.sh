# !/bin/bash

# This script converts videos into frames
# for different fps change (-r 1)

for f in *.avi
  do g=`echo $f | sed 's/\.avi//'`;
  echo Processing $f; 
  mkdir -p frames/$g/ ;
  ffmpeg -i $f frames/$g/image-%04d.jpeg ; 
done
