#! /bin/bash

database_dir=$1

for (( c=1; c<=100; c++))
do
	echo $database_dir
	python train_ens_w_sa.py $database_dir 2
	python train_ens_w_sa.py $database_dir .99
	python train_ens_w_sa_wtd.py $database_dir 2
	python train_ens_w_sa_wtd.py $database_dir .99
	rm $database_dir'cv_cur' $database_dir/new_wts $database_dir/weight_file $database_dir/new_wts_dev
done
