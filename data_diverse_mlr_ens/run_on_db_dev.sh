#! /bin/bash

database_dir=$1

for (( c=1; c<=100; c++))
do
	echo $database_dir
	python train_ens_w_sa_dev.py $database_dir 2
	python train_ens_w_sa_dev.py $database_dir .99
	rm cv_cur_dev new_wts_dev weight_file_dev 
done
