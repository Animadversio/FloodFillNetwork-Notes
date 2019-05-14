#PBS -N ffn-agglomeration-graph-p11_6-exp1-33-38-consensus
# 
# Specify the resources needed.  FreeSurfer just needs 1 core and
# 24 hours is usually enough.  This assumes the job requires less 
# than 3GB of memory.  If you increase the memory requested, it
# will limit the number of jobs you can run per node, so only  
# increase it when necessary (i.e. the job gets killed for violating
# the memory limit).

#PBS -l nodes=1:ppn=8,walltime=24:00:00,mem=10gb
# Request less resource
# Specify the default queue for the fastest nodes
#PBS -q old
#PBS -m be

source activate dlc_gpu
module load cuDNN-7.1.1
module load cuda-9.0p1
cd ~/FloodFillNetwork-Notes/
python3 agglomeration_graph_gen.py \
--config_path "/scratch/binxu.wang/results/LGN/p11_6_consensus_33_38_full/reseg_request.pbtxt" \
--output_dir "/scratch/binxu.wang/results/LGN/p11_6_consensus_33_38_full/agglomeration" \
--pixelsize '(8, 8, 40)' 