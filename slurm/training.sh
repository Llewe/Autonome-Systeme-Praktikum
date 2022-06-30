#!/bin/sh
source l-ki/bin/activate


cleanupSlurmOut()
{
 rm slurm/slurm-*.out
}

start()
{
  sbatch --partition=All --cpus-per-task=4 python3 ../main.py $@
}

# delete all old slurm logfiles
cleanupSlurmOut

# list of jobs to start
start 

#!/bin/sh
git fetch
git switch slurm
git pull

# Create python enviroment
python3 -m venv l-ki
source l-ki/bin/activate

# Install dependencies
pip3 install -r ../requirements.txt

# Fix permissions for unity enviroments
chmod -R 755 ../unity-env/*.x86_64

# Start skript to start all training processes
sh jobList.sh