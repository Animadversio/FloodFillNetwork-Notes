

#PBS -N ffn-resegment-p11_6-exp1-33-38-consensus-array
# not very fast, must parallelize 43 pairs / 6min
# Specify the resources needed.  FreeSurfer just needs 1 core and
# 24 hours is usually enough.  This assumes the job requires less 
# than 3GB of memory.  If you increase the memory requested, it
# will limit the number of jobs you can run per node, so only  
# increase it when necessary (i.e. the job gets killed for violating
# the memory limit).

#PBS -l nodes=1:ppn=2:haswell:gpus=1,walltime=24:00:00,mem=10gb
# Request less resource
# Specify the default queue for the fastest nodes
#PBS -q dque
#PBS -m be
#PBS -t 1-45

cd /scratch/binxu.wang/results/LGN/p11_6_consensus_33_38_full/

file_list=`ls /scratch/binxu.wang/results/LGN/p11_6_consensus_33_38_full/reseg_points/*.txt` 
point_path="$(echo "$file_list" | head -n $PBS_ARRAYID | tail -1)"
export config='inference {
    image {
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
    }
    init_segmentation {
       npz: "/scratch/binxu.wang/results/LGN/p11_6_consensus_33_38_full/0/0/seg-0_0_0.npz:segmentation"
    }
}
radius {x: 50 y: 50 z: 17}
output_directory: "/scratch/binxu.wang/results/LGN/p11_6_consensus_33_38_full/reseg"
max_retry_iters: 10
segment_recovery_fraction: 0.6
analysis_radius {x: 35 y: 35 z: 10}
'
# export point_list=`cat resegment_point_list.txt`

echo "$config" > reseg_request.pbtxt
# awk 'NR >= 400 && NR <= 5000' resegment_point_list.txt >> request.pbtxt # select first 5000 seeds into request
# cat tmp.txt resegment_point_list.txt > request.pbtxt
# rm tmp.txt 
cd ~/FloodFillNetwork-Notes/
export HDF5_USE_FILE_LOCKING=FALSE
module load singularity-2.4.2
singularity exec -B /scratch:/scratch --nv /export/tensorflow/tensorflow_1.7.0_gpu python3 run_resegment.py \
  --config "$config" \
  --point_path "$point_path" \
  --pixelsize '(8, 8, 40)' 