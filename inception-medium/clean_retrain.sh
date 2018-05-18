rm -rf bottleneck/multi-label
mkdir bottleneck
mkdir bottleneck/multi-label
rm results.txt
rm labels.txt
rm labels_count.txt
rm -rf image_labels_dir
mkdir image_labels_dir
rm -rf images
mkdir images
mkdir images/multi-label
python getYlabels.py
python reduce_images.py
./retrain.sh
