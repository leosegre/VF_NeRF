import os
import subprocess
from datetime import datetime
import sys
import numpy as np

import json

import random


def main(data_dir, outputs_dir, scene_names, downscale, timestamp=None, repeat_reg=10):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        reconstruct_scenes = True
    else:
        reconstruct_scenes = False

    print(timestamp)

    default_params = "ns-train nerfacto --viewer.quit-on-train-completion True --max-num-iterations 60000 --nf-first-iter 50000 --pipeline.datamanager.train-num-rays-per-batch 1024 " \
                     "--pipeline.model.predict-view-likelihood True --pipeline.datamanager.camera-optimizer.mode off --vis viewer+tensorboard "
    default_params_registered = " nerfstudio-data --auto_scale_poses True --train-split-fraction 1.0 --center-method focus --orientation-method up --scene-scale 2 "
    default_params_unregistered = " nerfstudio-data --train-split-fraction 1.0 --max-translation 0.0 --max-angle-factor 0.0 --scene-scale 2 " \
                                  "--registration True --orientation-method none --center-method none --auto-scale-poses False "
    default_params_registration = "ns-train register-nerfacto --viewer.quit-on-train-completion True --pipeline.model.predict-view-likelihood True --nf-first-iter 100000 " \
                                  "--start-step 0 --pipeline.datamanager.train-num-rays-per-batch 32000 --max-num-iterations 5000 --downscale_init 4 " \
                                  "--pipeline.model.distortion-loss-mult 0 --pipeline.model.interlevel-loss-mult 0 --pipeline.registration True --vis viewer+tensorboard " \
                                  "--pipeline.datamanager.camera-optimizer.optimizer.lr 5e-3 --pipeline.datamanager.camera-optimizer.scheduler.lr-final 5e-4"
    default_params_registration_suffix = " nerfstudio-data --train-split-fraction 1.0 --max-translation 0.5 --max-angle-factor 0.25 --scene-scale 2 --registration True " \
                                         "--optimize_camera_registration True --load_registration True --orientation-method none --center-method none --auto-scale-poses False "

    print(scene_names)

    exps = []

    for scene in scene_names:
        exp_params = {
            "data1": f"{data_dir}/{scene}/transforms.json",
            "data2": f"{data_dir}/{scene}_light/transforms.json",
            "experiment_name": f"{scene}",
            "scene_name": f"{scene}",
            "downscale_factor": f"{downscale}",
            "reg_downscale_factor": f"{int(downscale)}",
            "num_points_reg": "10",
            "num_points_unreg": "10",
            "pretrain-iters": "25",
            "unreg_data_dir": f"{data_dir}/",
            "outputs_dir": f"{outputs_dir}"
        }
        exps.append(exp_params)

    total_stats = {}
    for exp in exps:
        print("experiment_name:", exp["experiment_name"])
        registered_scene_cmd = default_params + "--output-dir " + exp["outputs_dir"] + " --machine.seed {}"\
                               " --data " + exp["data1"] + " --experiment_name " + exp["experiment_name"]  \
                               + "_registered --timestamp " + timestamp + default_params_registered + \
                               "--downscale_factor " + exp["downscale_factor"]

        unregistered_scene_cmd = default_params + "--output-dir " + exp["outputs_dir"] + " --machine.seed {}" \
                               " --data " + exp["data2"] + " --experiment_name " + exp["experiment_name"]  \
                               + "_unregistered --timestamp " + timestamp + default_params_unregistered + \
                               "--downscale_factor " + exp["downscale_factor"] + \
                               " --registration_data " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp

        export_cmd_unreg = "ns-export nf-cameras --seed {} --load-config " + outputs_dir + exp["experiment_name"] + "_unregistered/nerfacto/" \
                     + timestamp + "/config.yml" + " --output-dir " + exp["unreg_data_dir"] + exp["experiment_name"] + "_unregistered" \
                     + " --num_points " + exp["num_points_unreg"] + " --downscale_factor " + exp["reg_downscale_factor"] \
                     + " --sample_ratio 100"

        registeration_cmd = default_params_registration + " --output-dir " + exp["outputs_dir"] + \
                            " --pretrain-iters " + exp["pretrain-iters"] + " --machine.seed {}" \
                            + " --data " + exp["unreg_data_dir"] + exp["experiment_name"] + "_unregistered" \
                            + " --experiment_name " + exp["experiment_name"] + "_registration --timestamp " + timestamp \
                            + " --load_dir " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp + "/nerfstudio_models/" \
                            + default_params_registration_suffix + " --downscale_factor " + exp["reg_downscale_factor"]

        scene_seed = np.array(list(exp["scene_name"].encode('ascii'))).sum()
        stats_list = []

        if reconstruct_scenes:
            os.system(registered_scene_cmd.format(scene_seed))
            os.system(unregistered_scene_cmd.format(scene_seed))

        best_psnr = 0
        for i in range(1, repeat_reg+1):
            os.system(export_cmd_unreg.format(str(scene_seed*i)))
            os.system(registeration_cmd.format(str(scene_seed*i)))

            # Read the stats of the registration
            exp_stats_path = outputs_dir + exp["experiment_name"] + "_registration/nerfacto/" + timestamp + "/stats.json"
            with open(os.path.join(exp_stats_path), 'r') as f:
                exp_stats = json.load(f)
            stats_list.append(exp_stats)
            if exp_stats["psnr"] > best_psnr:
                best_psnr = exp_stats["psnr"]
                best_exp_stats = exp_stats

        total_stats[exp["experiment_name"]] = {"best": best_exp_stats, "stats_list": stats_list}

    curr_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    base_dir = f"{outputs_dir}/../stats/"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    total_stats_path = base_dir + curr_timestamp + str(random.randint(0, 100)) + "_light.json"
    with open(total_stats_path, "w") as outfile:
        json.dump(total_stats, outfile, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("Usage: python reg_pipeline.py <data_directory> <output_directory> <scene_names> <exp_types> <downscale> <<timestamp>>")
    else:
        base_directory = sys.argv[1]
        output_directory = sys.argv[2]
        scene_names = sys.argv[3].split(',')
        downscale = sys.argv[4]

        if not os.path.isdir(base_directory):
            print(f"Error: {base_directory} is not a valid directory.")
        elif not os.path.isdir(output_directory):
            print(f"Error: {output_directory} is not a valid directory.")
        else:
            if len(sys.argv) == 6:
                timestamp = sys.argv[5]
                main(base_directory, output_directory, scene_names, downscale, timestamp)
            else:
                main(base_directory, output_directory, scene_names, downscale)




