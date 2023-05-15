#salloc --nodes=1 --ntasks=1 --mem=8G --time=02:00:00 --gres=gpu:1

export XDG_RUNTIME_DIR=""

#module load cudnn/cuda-11.x/8.2.0
source activate keypoint_moseq_sleap

# get tunneling info
node=$(hostname -s)
user=$(whoami)
cluster="della"
port=8889

# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}.princeton.edu

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"
jupyter-notebook --no-browser --port=${port} --ip=${node}
