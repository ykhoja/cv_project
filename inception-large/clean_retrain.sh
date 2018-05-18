# clean_retrain.sh makes sure no files from previous runs will be around

# bottlenecks might take very long to create, so if there are no new images, consider using (not deleting) them

rm -rf bottlenecks/multi-label
mkdir bottlenecks
mkdir bottlenecks/multi-label

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
