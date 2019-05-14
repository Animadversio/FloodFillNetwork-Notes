#!/bin/sh

#PBS -N ffn-inference-Longterm-Mov-p11_6-exp1-38-array
# Pretest of the ability of new model on new aligned volume

#PBS -l nodes=1:ppn=2:haswell:gpus=1,walltime=24:00:00,mem=15gb
# Request less resource
# Specify the default queue for the fastest nodes
#PBS -m be
#PBS -q dque
#PBS -t 36-45

# Prepare the virtual env for python
# export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
# source activate conda_env

param_list='start { x:1000 y:   0 z:0 } size { x:600 y:600 z:152 }
start { x:1500 y:   0 z:0 } size { x:600 y:600 z:152 }
start { x:2000 y:   0 z:0 } size { x:600 y:600 z:152 }
start { x:2500 y:   0 z:0 } size { x:600 y:600 z:152 }
start { x:1000 y: 500 z:0 } size { x:600 y:600 z:152 }
start { x:1500 y: 500 z:0 } size { x:600 y:600 z:152 }
start { x:2000 y: 500 z:0 } size { x:600 y:600 z:152 }
start { x:2500 y: 500 z:0 } size { x:600 y:600 z:152 }
start { x:1000 y:1000 z:0 } size { x:600 y:600 z:152 }
start { x:1500 y:1000 z:0 } size { x:600 y:600 z:152 }
start { x:2000 y:1000 z:0 } size { x:600 y:600 z:152 }
start { x:2500 y:1000 z:0 } size { x:600 y:600 z:152 }
start { x:1000 y:1500 z:0 } size { x:600 y:600 z:152 }
start { x:1500 y:1500 z:0 } size { x:600 y:600 z:152 }
start { x:2000 y:1500 z:0 } size { x:600 y:600 z:152 }
start { x:2500 y:1500 z:0 } size { x:600 y:600 z:152 }
start { x:1000 y:2000 z:0 } size { x:600 y:600 z:152 }
start { x:1500 y:2000 z:0 } size { x:600 y:600 z:152 }
start { x:2000 y:2000 z:0 } size { x:600 y:600 z:152 }
start { x:2500 y:2000 z:0 } size { x:600 y:600 z:152 }
start { x:   0 y:   0 z:0 } size { x:600 y:600 z:152 }
start { x: 500 y:   0 z:0 } size { x:600 y:600 z:152 }
start { x:3000 y:   0 z:0 } size { x:600 y:600 z:152 }
start { x:3500 y:   0 z:0 } size { x:600 y:600 z:152 }
start { x:4000 y:   0 z:0 } size { x:600 y:600 z:152 }
start { x:   0 y: 500 z:0 } size { x:600 y:600 z:152 }
start { x: 500 y: 500 z:0 } size { x:600 y:600 z:152 }
start { x:3000 y: 500 z:0 } size { x:600 y:600 z:152 }
start { x:3500 y: 500 z:0 } size { x:600 y:600 z:152 }
start { x:4000 y: 500 z:0 } size { x:600 y:600 z:152 }
start { x:   0 y:1000 z:0 } size { x:600 y:600 z:152 }
start { x: 500 y:1000 z:0 } size { x:600 y:600 z:152 }
start { x:3000 y:1000 z:0 } size { x:600 y:600 z:152 }
start { x:3500 y:1000 z:0 } size { x:600 y:600 z:152 }
start { x:4000 y:1000 z:0 } size { x:600 y:600 z:152 }
start { x:   0 y:1500 z:0 } size { x:600 y:600 z:152 }
start { x: 500 y:1500 z:0 } size { x:600 y:600 z:152 }
start { x:3000 y:1500 z:0 } size { x:600 y:600 z:152 }
start { x:3500 y:1500 z:0 } size { x:600 y:600 z:152 }
start { x:4000 y:1500 z:0 } size { x:600 y:600 z:152 }
start { x:   0 y:2000 z:0 } size { x:600 y:600 z:152 }
start { x: 500 y:2000 z:0 } size { x:600 y:600 z:152 }
start { x:3000 y:2000 z:0 } size { x:600 y:600 z:152 }
start { x:3500 y:2000 z:0 } size { x:600 y:600 z:152 }
start { x:4000 y:2000 z:0 } size { x:600 y:600 z:152 }'

echo "$param_list" | head -n $PBS_ARRAYID | tail -1
# read the input information in txt file in input
input="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"
# Note the syntax and no speccial character! 
# `head -n $PBS_ARRAYID  param_list.txt | tail -1`

# cd Into the run directory; I'll create a new directory to run under
# Make the request file 
export logfile_name="inference_p11_6_exp1-38_batch_${PBS_ARRAYID}.log"
export bounding_box=$input
export Request='image {
  hdf5: "/scratch/binxu.wang/ffn-Data/LGN_DATA/grayscale_ixP11_6_align_norm.h5:raw"
}
image_mean: 138
image_stddev: 38
checkpoint_interval: 1800
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/scratch/binxu.wang/ffn-Data/models/LR_model_Longtime_SF_Deep/model.ckpt-15392756"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 12, \"fov_size\": [33, 33, 17], \"deltas\": [8, 8, 4]}"
segmentation_output_dir: "/scratch/binxu.wang/results/LGN/p11_6_exp1-38"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 5000
  disco_seed_threshold: 0.002
}'

# Close the FILE lock system ! to use hdf5 on this machine 
export HDF5_USE_FILE_LOCKING=FALSE
echo $bounding_box
# Finally, run the command (mount the scratch disk)

cd ~/FloodFillNetwork-Notes/
module load singularity-2.4.2
singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python3 ~/FloodFillNetwork-Notes/run_inference.py \
  --inference_request="$Request" \
  --bounding_box "$bounding_box" \
  --logfile_name "$logfile_name"

singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python3 visualize_segmentation_script.py \
  --inference_request="$Request" \
  --stat_only True \
  --bounding_box "$bounding_box" 
 
