#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N ffn-train-WF-Longtime3
#PBS -m be
#PBS -l nodes=1:ppn=4:gpus=1:V100,walltime=168:00:00,mem=6gb

# Specify the default queue for the fastest nodes
#PBS -q dque

cd ~/ffn-master/

# Close the FILE lock system ! to use hdf5 on this machine 
export HDF5_USE_FILE_LOCKING=FALSE
# Finally, run the command (mount the scratch disk)
module load singularity-2.4.2
singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python ~/ffn-master/train.py \
--train_coords /scratch/binxu.wang/ffn-Data/LGN_DATA/tf_record_file_LR_WF \
--data_volumes LGN_LR:/scratch/binxu.wang/ffn-Data/LGN_DATA/grayscale_maps_LR.h5:raw \
--label_volumes LGN_LR:/scratch/binxu.wang/ffn-Data/LGN_DATA/groundtruth_LR.h5:stack \
--train_dir /scratch/binxu.wang/ffn-Data/models/LR_model_WF_Longtime \
--model_name convstack_3d.ConvStack3DFFNModel \
--model_args "{\"depth\": 9, \"fov_size\": [77, 51, 23], \"deltas\": [15,10,5]}" \
--max_steps 30000000 \
--summary_rate_secs 300 \
--image_mean 136 \
--image_stddev 55 \
--permutable_axes 0

# image mean 135.86
# image std 54.45