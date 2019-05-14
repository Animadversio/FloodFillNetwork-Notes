#!/bin/sh

#PBS -N ffn-inference-Longterm-Mov-p11_6-exp1-array
# Pretest of the ability of new model on new aligned volume

#PBS -l nodes=1:ppn=2:haswell:gpus=1,walltime=24:00:00,mem=15gb
# Request less resource
# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 32-45

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env

# param_list='start { x:0 y:0 z:0 } size { x:700 y:700 z:152 }
# start { x:500 y:0 z:0 } size { x:700 y:700 z:152 }
# start { x:1000 y:0 z:0 } size { x:700 y:700 z:152 }
# start { x:0 y:500 z:0 } size { x:700 y:700 z:152 }
# start { x:500 y:500 z:0 } size { x:700 y:700 z:152 }
# start { x:1000 y:500 z:0 } size { x:700 y:700 z:152 }
# start { x:0 y:1000 z:0 } size { x:700 y:700 z:152 }
# start { x:500 y:1000 z:0 } size { x:700 y:700 z:152 }
# start { x:1000 y:1000 z:0 } size { x:700 y:700 z:152 }'

param_list='Longtime_Mov/model.ckpt-14204266
Longtime_SF_Deep/model.ckpt-15198901
Longtime_SF_Deep/model.ckpt-14505407
Longtime_SF_Deep/model.ckpt-14005282
Longtime_SF_Deep/model.ckpt-13505286
Longtime_SF_Deep/model.ckpt-13070367
Longtime_SF_Deep/model.ckpt-12506484
Longtime_SF_Deep/model.ckpt-11564900
Longtime_SF_Deep/model.ckpt-10057126
Longtime_SF_Deep/model.ckpt-8506460
Longtime/model.ckpt-13150773
Longtime/model.ckpt-12509612
Longtime/model.ckpt-12055073
Longtime/model.ckpt-11502735
Longtime/model.ckpt-11001149
Longtime/model.ckpt-10508695
Longtime/model.ckpt-10007510
Longtime/model.ckpt-9006720
Longtime/model.ckpt-8023383
Longtime_Mov/model.ckpt-25353276
Longtime_Mov/model.ckpt-24094971
Longtime_Mov/model.ckpt-23507478
Longtime_Mov/model.ckpt-23007150
Longtime_Mov/model.ckpt-22506165
Longtime_Mov/model.ckpt-22008759
Longtime_Mov/model.ckpt-21004393
Longtime_Mov/model.ckpt-20001235
Longtime_Mov/model.ckpt-19006308
Longtime_Mov/model.ckpt-18003209
Longtime_Mov/model.ckpt-17005779
Longtime_SF_Deep/model.ckpt-15623522
Longtime_SF_Deep/model.ckpt-16128269
Longtime_SF_Deep/model.ckpt-15005144
Longtime_SF_Deep/model.ckpt-15097418
Longtime_SF_Deep/model.ckpt-15254261
Longtime_SF_Deep/model.ckpt-15152786
Longtime_SF_Deep/model.ckpt-15309650
Longtime_SF_Deep/model.ckpt-15392756
Longtime_SF_Deep/model.ckpt-15448124
Longtime_SF_Deep/model.ckpt-15503491
Longtime_SF_Deep/model.ckpt-15697394
Longtime_SF_Deep/model.ckpt-15795858
Longtime_SF_Deep/model.ckpt-15897387
Longtime_SF_Deep/model.ckpt-15998950
Longtime_SF_Deep/model.ckpt-16063625'

echo "$param_list" | head -n 1 | tail -1
# read the input information in txt file in input
# Note the syntax and no speccial character! 
# `head -n $PBS_ARRAYID  param_list.txt | tail -1`
export model_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
export logfile_name="inference_p11_6_exp1_batch_${PBS_ARRAYID}.log"


# cd Into the run directory; I'll create a new directory to run under
# Make the request file 
export bounding_box='start { x: 500 y: 500 z:0 } size { x:600 y:600 z:152 }'
export bounding_box1='start { x: 1500 y:  0 z:0 } size { x:600 y:600 z:152 }'

# cd Into the run directory; I'll create a new directory to run under
# Make the request file 
# model_name=Longtime_Mov/model.ckpt-17907771
export Request='image {
  hdf5: "/scratch/binxu.wang/ffn-Data/LGN_DATA/grayscale_ixP11_6_align_norm.h5:raw"
}
image_mean: 138
image_stddev: 38
checkpoint_interval: 1800
seed_policy: "PolicyPeaks"
model_name: "convstack_3d.ConvStack3DFFNModel"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 5000
  disco_seed_threshold: 0.002
}'

model_ckpt_path="model_checkpoint_path: \"/scratch/binxu.wang/ffn-Data/models/LR_model_${model_name}\""
if [[ $model_name == *"Mov"* ]]; then
model_args='model_args: "{\"depth\": 9, \"fov_size\": [37, 25, 15], \"deltas\": [8,6,2]}"'
elif [[ $model_name == *"Deep"* ]]; then
model_args='model_args: "{\"depth\": 12, \"fov_size\": [33, 33, 17], \"deltas\": [8, 8, 4]}"' 
else
model_args='model_args: "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}"'
fi
output_dir="segmentation_output_dir: \"/scratch/binxu.wang/results/LGN/p11_6_exp1-${PBS_ARRAYID}\""

Request+=$'\n'
Request+="$model_ckpt_path"
Request+=$'\n'
Request+="$model_args"
Request+=$'\n'
Request+="$output_dir"

echo "$model_name"
echo "$model_args"
echo "$Request"
echo "$bounding_box"
echo "$logfile_name"

# Close the FILE lock system ! to use hdf5 on this machine 
export HDF5_USE_FILE_LOCKING=FALSE
cd ~/FloodFillNetwork-Notes/
# Finally, run the command (mount the scratch disk)
module load singularity-2.4.2
singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python3 ~/FloodFillNetwork-Notes/run_inference.py \
  --inference_request="$Request" \
  --bounding_box "$bounding_box" \
  --logfile_name "$logfile_name"

singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python3 ~/FloodFillNetwork-Notes/run_inference.py \
  --inference_request="$Request" \
  --bounding_box "$bounding_box1" \
  --logfile_name "$logfile_name"


singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python3 visualize_segmentation_script.py \
  --inference_request="$Request" \
  --stat_only True \
  --bounding_box "$bounding_box" 

singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python3 visualize_segmentation_script.py \
  --inference_request="$Request" \
  --stat_only True \
  --bounding_box "$bounding_box1" 
 