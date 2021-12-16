from runset_train import parameters
from runset_train import get_input_runsets_crt_opts
import os
from subprocess import call

def main(opts_strs, params_dict):
    slurm_submit_dir = "/scratch/groups/gracegao/" + params_dict.proj_name + "/job_files/"
    code_dir = ""
    job_script_name = os.path.join(slurm_submit_dir, "latest_job_sc.sh")
    if not os.path.exists(slurm_submit_dir):
        try:
            os.makedirs(slurm_submit_dir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    for no, opts in enumerate(opts_strs):
        args = parameters.get_arguments(params_dict, opts_str=opts)
        repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos])
        #print(repr_str)
        
        waitings_folder = os.path.join("/scratch/groups/gracegao/" + params_dict.proj_name + "/waiting_jobs", repr_str)
        if not os.path.exists(waitings_folder):
            try:
                os.makedirs(waitings_folder)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        
        
        job_time = 40 if args.ini in ["sp1","sp2"] else 25
        
        run_commands = []
        if not (args.ini in ["sp1","sp2"]):
            run_commands.append("python3 -m runset_train.train{}\n".format(opts))
        else:
            last_N = args.N
            curr_N = min(8, last_N)
            while curr_N <= last_N:
                args.N = curr_N
                curr_opts = get_input_runsets_crt_opts.create_opts_strs([args], params_dict).split("\n")[0]
                run_commands.append("python3 -m runset_train.train{}\n".format(curr_opts))
                curr_N *= 2
                
        script = ""
        script += "#!/bin/bash\n"
        script += "#SBATCH --time={}:00:00\n".format(job_time)
        script += "#SBATCH --job-name={}\n".format(repr_str)
        script += "#SBATCH --mail-user=type_your_mail@stanford.edu\n"
        script += "#SBATCH --mail-type=BEGIN,END\n" # mail on beginning and end
        
        script +=  "#SBATCH --output={}%j.%x.out\n".format(slurm_submit_dir)
        script +=  "#SBATCH --error={}%j.%x.err\n".format(slurm_submit_dir)

        script += "#SBATCH --nodes=1\n"
        script += "#SBATCH --ntasks-per-node=1\n"
        script += "#SBATCH --partition=gpu\n"
        script += "#SBATCH --gpus=1\n"
        #script += "#SBATCH -C GPU_MEM:32GB" if needed
        
        script += "module load python/3.9.0\n"
        script += "module load py-numpy/1.20.3_py39\n"
        script += "module load py-pytorch/1.8.1_py39\n"
        script += "module load py-matplotlib/3.4.2_py39\n"
        script += "module load cuda/11.2.0\n"

        script += "python3 -m runset_train.remove_from_waitings{}\n".format(opts)
        #script += 'nvidia-smi' if needed
        for command in run_commands:
            script += command 
        job_script_file = open(job_script_name, "w+")
        job_script_file.write(script)
        job_script_file.close()
        rc = call("sbatch {}".format(job_script_name), shell=True)
if __name__ == "__main__":
    main()
