cleanupSlurmOut()
{
 rm slurm-*.out
}

startTraining()
{
  sbatch --partition=All --cpus-per-task=4 startPyMain.sh $@
}
cleanupSlurmOut

startTraining \
-tag test \
-env unity \
-env_n 3DBall1 \
-agent ppo \
-e 10 \
-us 10 \
-g 0.99 \
-lr_a 1e-03 \
-lr_c 2e-03 \
-ke 15 \
-e_clip 0.3 \
-a_std 0.8 \
-a_std_rate 5e-4 \
-a_std_freq 1000 \
-a_std_min 1e-3