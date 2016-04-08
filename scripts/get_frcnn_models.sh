#!/bin/bash
dir_name=${PWD##*/}
echo $dir_name

file_name=frcnn_torch_models.tar.gz
URL=http://www.umiacs.umd.edu/~najibi/data/frcnn_torch_models.tar.gz


if [ "$dir_name" = "scripts" ]; then
	save_path=../data/trained_models
else
	save_path=./data/trained_models
fi


file_path=${save_path%%/}/$file_name


if [ -f "$file_path" ]; then
	echo "File already exits!"
else
	if [ ! -d "$save_path" ]; then
		mkdir $save_path
	fi
	echo "Downloading the fast-rcnn_models..."
	wget $URL -O $file_path
	echo "Uzip the file..."

	tar zxvf $file_path -C $save_path
	echo "Done!"
fi
