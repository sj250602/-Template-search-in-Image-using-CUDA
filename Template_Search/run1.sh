cd $PBS_O_WORKDIR
module add compiler/cuda/11.0/compilervars
./a4 ~/scratch/A4/test4/data_image.txt ~/scratch/A4/test4/query_image.txt 12 8 5