import submitit

def main():
    # function = submitit.helpers.CommandFunction(["module", "load", "Miniconda3"])
    executor = submitit.AutoExecutor(folder="logs", slurm_python='~/.conda/envs/torch/bin/python')
    executor.update_parameters(
            mem_gb=100,
            gpus_per_node=1,
            tasks_per_node=1,  # one task per GPU
            cpus_per_task=64,
            timeout_min=60*1,  # max is 60 * 72
            )

if __name__ == '__main__':
    main()