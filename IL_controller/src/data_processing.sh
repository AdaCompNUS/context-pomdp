data_path=$1
dataset_path=$2
echo "python3 Data_processing/parallel_parse_pool.py --bagspath $data_path"
python3 Data_processing/parallel_parse_pool.py --bagspath $data_path
# echo "./h5/*.h5"
# rm h5/*.h5
# echo "$data_path -type f -name '*.h5' -exec mv -i {} ./h5/  \;"
# find $data_path -type f -name '*.h5' -exec mv -i {} ./h5/  \;
rm $dataset_path/train.h5
rm $dataset_path/val.h5
rm $dataset_path/test.h5
echo "Data_processing/combine.py --bagspath $data_path --samplemode random --outpath $dataset_path/"
python3 Data_processing/combine.py --bagspath $data_path --samplemode random --outpath $dataset_path/