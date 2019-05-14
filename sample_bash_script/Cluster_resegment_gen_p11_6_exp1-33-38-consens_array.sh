#!/bin/sh

# give the job a name to help keep track of running jobs (optional)
#PBS -N ffn-resegment-gen-p11_6-exp1-33-38-consensus
#PBS -m be
#PBS -l nodes=1:ppn=8,walltime=1:00:00,mem=10gb
#PBS -t 1-45
# Specify the default queue for the fastest nodes
#PBS -q old

export PATH=/act/Anaconda3-2.3.0/bin:${PATH}
source activate conda_env

param_list='start { x:1000 y:   0 z:0 } size { x:550 y:550 z:152 }
start { x:1500 y:   0 z:0 } size { x:550 y:550 z:152 }
start { x:2000 y:   0 z:0 } size { x:550 y:550 z:152 }
start { x:2500 y:   0 z:0 } size { x:550 y:550 z:152 }
start { x:1000 y: 500 z:0 } size { x:550 y:550 z:152 }
start { x:1500 y: 500 z:0 } size { x:550 y:550 z:152 }
start { x:2000 y: 500 z:0 } size { x:550 y:550 z:152 }
start { x:2500 y: 500 z:0 } size { x:550 y:550 z:152 }
start { x:1000 y:1000 z:0 } size { x:550 y:550 z:152 }
start { x:1500 y:1000 z:0 } size { x:550 y:550 z:152 }
start { x:2000 y:1000 z:0 } size { x:550 y:550 z:152 }
start { x:2500 y:1000 z:0 } size { x:550 y:550 z:152 }
start { x:1000 y:1500 z:0 } size { x:550 y:550 z:152 }
start { x:1500 y:1500 z:0 } size { x:550 y:550 z:152 }
start { x:2000 y:1500 z:0 } size { x:550 y:550 z:152 }
start { x:2500 y:1500 z:0 } size { x:550 y:550 z:152 }
start { x:1000 y:2000 z:0 } size { x:550 y:550 z:152 }
start { x:1500 y:2000 z:0 } size { x:550 y:550 z:152 }
start { x:2000 y:2000 z:0 } size { x:550 y:550 z:152 }
start { x:2500 y:2000 z:0 } size { x:550 y:550 z:152 }
start { x:   0 y:   0 z:0 } size { x:550 y:550 z:152 }
start { x: 500 y:   0 z:0 } size { x:550 y:550 z:152 }
start { x:3000 y:   0 z:0 } size { x:550 y:550 z:152 }
start { x:3500 y:   0 z:0 } size { x:550 y:550 z:152 }
start { x:4000 y:   0 z:0 } size { x:550 y:550 z:152 }
start { x:   0 y: 500 z:0 } size { x:550 y:550 z:152 }
start { x: 500 y: 500 z:0 } size { x:550 y:550 z:152 }
start { x:3000 y: 500 z:0 } size { x:550 y:550 z:152 }
start { x:3500 y: 500 z:0 } size { x:550 y:550 z:152 }
start { x:4000 y: 500 z:0 } size { x:550 y:550 z:152 }
start { x:   0 y:1000 z:0 } size { x:550 y:550 z:152 }
start { x: 500 y:1000 z:0 } size { x:550 y:550 z:152 }
start { x:3000 y:1000 z:0 } size { x:550 y:550 z:152 }
start { x:3500 y:1000 z:0 } size { x:550 y:550 z:152 }
start { x:4000 y:1000 z:0 } size { x:550 y:550 z:152 }
start { x:   0 y:1500 z:0 } size { x:550 y:550 z:152 }
start { x: 500 y:1500 z:0 } size { x:550 y:550 z:152 }
start { x:3000 y:1500 z:0 } size { x:550 y:550 z:152 }
start { x:3500 y:1500 z:0 } size { x:550 y:550 z:152 }
start { x:4000 y:1500 z:0 } size { x:550 y:550 z:152 }
start { x:   0 y:2000 z:0 } size { x:550 y:550 z:152 }
start { x: 500 y:2000 z:0 } size { x:550 y:550 z:152 }
start { x:3000 y:2000 z:0 } size { x:550 y:550 z:152 }
start { x:3500 y:2000 z:0 } size { x:550 y:550 z:152 }
start { x:4000 y:2000 z:0 } size { x:550 y:550 z:152 }'

echo "$param_list" | head -n $PBS_ARRAYID | tail -1
# read the input information in txt file in input
export bounding_box="$(echo "$param_list" | head -n $PBS_ARRAYID | tail -1)"

cd ~/FloodFillNetwork-Notes/
# Close the FILE lock system ! to use hdf5 on this machine 
export HDF5_USE_FILE_LOCKING=FALSE
# Finally, run the command (mount the scratch disk)
# module load singularity-2.4.2
# singularity exec -B /scratch:/scratch --nv /export/tensorflow/tensorflow_1.7.0_gpu 
python3 resegment_seed_generation_Center_Mass.py \
--seg_path "/scratch/binxu.wang/results/LGN/p11_6_consensus_33_38_full" \
--output_path "/scratch/binxu.wang/results/LGN/p11_6_consensus_33_38_full/reseg_points" \
--corner  '(0,0,0)' \
--bounding_box "$bounding_box" 
# parallelized to 45 jobs finished within 14 min on 45 old nodes 
# \"start { x:3500 y:2000 z:0 } size { x:550 y:550 z:152 }"
# --offset "$offset" \
# --size '(152, 550, 550)' 

# /scratch/binxu.wang/ffn-Data/results/LGN/testing_LR/0/0/ /scratch/binxu.wang/ffn-Data/results/LGN/testing_LR/


