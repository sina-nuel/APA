import os
import json
import subprocess
import torch


input_file = "./hparams.json"
with open(input_file, 'r') as f:
    hparams = json.load(f)

base_command = "python -m domainbed.scripts.train --data_dir=./domainbed/data/ --algorithm APA --dataset PACS"

output_base_dir = "./results"
os.makedirs(output_base_dir, exist_ok=True)



env_list = [0,1,2,3]
augStrt = 180
stp = 200 
nrm = 0.9
hparams['augStart'] = augStrt
hparams['mixStep'] = stp
hparams['normStep'] = int(stp*nrm)
output_dir = os.path.join(output_base_dir, f"stp_{stp}_nrm_{nrm}_strt_{strt}")
seed = 1    
output_dir = os.path.join(output_dir, f"seed_{seed}")
os.makedirs(output_dir, exist_ok=True)

for j in env_list:  # Adjust range based on the number of test environments
    torch.cuda.empty_cache()
    tmp_dir = os.path.join(output_dir, f"env_{j}")
    os.makedirs(tmp_dir, exist_ok=True)

    test_envs = [j]  
    test_envs_str = " ".join(map(str, test_envs))

    command = (
        f"{base_command} "
        f"--hparams='{json.dumps(hparams)}' "
        f"--output_dir={tmp_dir} "
        f"--model_dir={tmp_dir} "
        f"--test_envs {test_envs_str} "
        f"--steps=6000 "
        f"--seed={seed}"
    )

    print(f"Running: {command}")
    subprocess.run(command, shell=True)

print("All runs completed.")
