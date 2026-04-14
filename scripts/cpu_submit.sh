#!/bin/bash
#SBATCH -J 3B-2-ivfpq-subsample10p
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH -p icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=50G
#SBATCH --time=7:00:00
#SBATCH --mail-type=ALL

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

#!#############################################################
#!#### Environment Setup ######################################
#!#############################################################

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/12.1

export PATH="$HOME/.local/bin:$PATH"

#! Work directory (i.e. where the job will run):
workdir="/home/yff23/repos/semantic-layer-skipping"

# --- LOCAL SOFTWARE PATHS & PYTHON ENV ---
cd $workdir

# Activate the uv virtual environment
echo "Activating uv environment..."
source .venv/bin/activate

# Verify python environment in the SLURM output log for debugging
echo "Checking Environment..."
python --version
which python

#! Full command to execute:
# -u means unbuffered output, which is useful for real-time logging in SLURM
#application="python -m main"
application="python -m main --model_name Qwen/Qwen2.5-3B-Instruct --checkpoint_start 4 --checkpoint_end 36 --checkpoint_step 4 --train_samples 10000 --run_ivfpq_conversion --subsample_fraction 0.1 --target_prefix batch_20260407_025540"
options=""

#! Are you using OpenMP? If so increase this safe value to no more than 128:
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
CMD="$application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo -e "\nExecuting command:\n==================\n$CMD\n"

# Run the python script
eval $CMD

echo "Job completed at: `date`"
