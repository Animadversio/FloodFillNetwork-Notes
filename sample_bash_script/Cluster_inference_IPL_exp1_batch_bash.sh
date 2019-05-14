#!/bin/sh

#PBS -N ffn-inference-screen-IPL-exp1-array
# Pretest of the ability of new model on new aligned volume

#PBS -l nodes=1:ppn=1:haswell:gpus=1,walltime=24:00:00,mem=15gb
# Request less resource
# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 17-41

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

param_list='Longtime_SF_Deep/model.ckpt-15005144
Longtime_Mov/model.ckpt-14204266
Longtime_SF_Deep/model.ckpt-23428086
Longtime_SF_Deep/model.ckpt-23051543
Longtime_SF_Deep/model.ckpt-22574076
Longtime_SF_Deep/model.ckpt-22069126
Longtime_SF_Deep/model.ckpt-21609977
Longtime_SF_Deep/model.ckpt-21105924
Longtime_SF_Deep/model.ckpt-20612043
Longtime/model.ckpt-19897111
Longtime/model.ckpt-19584227
Longtime/model.ckpt-19316007
Longtime/model.ckpt-19003498
Longtime/model.ckpt-18605539
Longtime/model.ckpt-18358893
Longtime_Mov/model.ckpt-36721860
Longtime/model.ckpt-18406961
Longtime/model.ckpt-18310227
Longtime/model.ckpt-18250571
Longtime/model.ckpt-18000537
Longtime/model.ckpt-18106612
Longtime/model.ckpt-18205100
Longtime/model.ckpt-17503903
Longtime/model.ckpt-17005309
Longtime_SF_Deep/model.ckpt-19006523
Longtime_SF_Deep/model.ckpt-19505569
Longtime_SF_Deep/model.ckpt-20003019
Longtime_SF_Deep/model.ckpt-18502188
Longtime_SF_Deep/model.ckpt-18012203
Longtime_SF_Deep/model.ckpt-17504236
Longtime_SF_Deep/model.ckpt-17005495
Longtime_SF_Deep/model.ckpt-16506900
Longtime_SF_Deep/model.ckpt-16008186
Longtime_SF_Deep/model.ckpt-15503491
Longtime_SF_Deep/model.ckpt-15208125
Longtime_SF_Deep/model.ckpt-15106644
Longtime_SF_Deep/model.ckpt-14903606
Longtime_SF_Deep/model.ckpt-14801758
Longtime_SF_Deep/model.ckpt-14709151
Longtime_SF_Deep/model.ckpt-14440581
Longtime_SF_Deep/model.ckpt-14005282'

echo "$param_list" | head -n 1 | tail -1
# read the input information in txt file in input
# Note the syntax and no speccial character! 
# `head -n $PBS_ARRAYID  param_list.txt | tail -1`
export model_name="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
export logfile_name="/home/binxu.wang/inference_IPL_exp1_batch_${PBS_ARRAYID}.log"

# cd Into the run directory; I'll create a new directory to run under
# Make the request file 
export bounding_box='start { x: 2500 y: 500 z:0 } size { x:1000 y:1000 z:78 }'
# export bounding_box1='start { x: 1500 y:  0 z:0 } size { x:600 y:600 z:400 }'

# cd Into the run directory; I'll create a new directory to run under
# Make the request file 
# seed_policy_args: "{ \"reverse\": 1}"
export Request='image {
  hdf5: "/scratch/binxu.wang/ffn-Data/Retina_Data/grayscale_ixQ_IPL_align_norm.h5:raw"
}
image_mean: 133
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
output_dir="segmentation_output_dir: \"/scratch/binxu.wang/results/IPL/IPL_exp1-${PBS_ARRAYID}\""

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
# cd ~/FloodFillNetwork-Notes/
# Finally, run the command (mount the scratch disk)
# module load singularity-2.4.2
# singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python3 ~/FloodFillNetwork-Notes/run_inference.py \
#   --inference_request="$Request" \
#   --bounding_box "$bounding_box" \
#   --logfile_name "$logfile_name"

# source activate dlc_gpu
# module load cuDNN-7.1.1
# module load cuda-9.0p1
cd ~/FloodFillNetwork-Notes/
module load singularity-2.4.2
singularity exec -B /scratch:/scratch --nv /export/tensorflow/tensorflow_1.7.0_gpu python3 run_inference.py \
  --inference_request="$Request" \
  --bounding_box "$bounding_box" \
  --logfile_name "$logfile_name"
# singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python3 ~/FloodFillNetwork-Notes/run_inference.py \
#   --inference_request="$Request" \
#   --bounding_box "$bounding_box1" \
#   --logfile_name "$logfile_name"


# singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu 
singularity exec -B /scratch:/scratch --nv /export/tensorflow/tensorflow_1.7.0_gpu python3  visualize_segmentation_script.py \
  --inference_request="$Request" \
  --stat_only True \
  --bounding_box "$bounding_box" 

# singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python3 visualize_segmentation_script.py \
#   --inference_request="$Request" \
#   --stat_only True \
#   --bounding_box "$bounding_box1" 
 