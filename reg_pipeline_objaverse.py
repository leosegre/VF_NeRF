import os
import subprocess
from datetime import datetime
import sys
import numpy as np

import json

import random


def main(data_dir, outputs_dir, scene_names=None, timestamp=None, repeat_reg=1):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        reconstruct_scenes = True
    else:
        reconstruct_scenes = False

    print(timestamp)

    if scene_names is None:
        scene_names = os.listdir(data_dir)
        print(scene_names)

    default_params = "ns-train objaverse-nerfacto --viewer.quit-on-train-completion True --max-num-iterations 20000 --nf-first-iter 15000 " \
                      "--pipeline.datamanager.sample-without-mask True --pipeline.model.nf-loss-on-mask-only True " \
                      "--pipeline.datamanager.train-num-rays-per-batch 1024 " \
                     "--pipeline.model.predict-view-likelihood True --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard "
    default_params_registered = " nerfstudio-data --train-split-fraction 1.0 --scene-scale 1.5 --objaverse True --orientation-method none --center-method none --auto-scale-poses False --alpha-color white "
    default_params_unregistered = " nerfstudio-data --train-split-fraction 1.0 --scene-scale 1.5 --objaverse True --orientation-method none --center-method none --auto-scale-poses False --alpha-color white "
    default_params_registration = "ns-train register-objaverse-nerfacto --viewer.quit-on-train-completion True --pipeline.model.predict-view-likelihood True --nf-first-iter 100000 " \
                                  "--start-step 0 --pipeline.datamanager.train-num-rays-per-batch 8128 --max-num-iterations 2500 --pipeline.objaverse True " \
                                  "--pipeline.model.distortion-loss-mult 0 --pipeline.model.interlevel-loss-mult 0 --pipeline.registration True --vis tensorboard" \
                                  " --pipeline.model.mse-init True "
    default_params_registration_suffix = " nerfstudio-data --train-split-fraction 1.0 --scene-scale 1.5 --registration True " \
                                         "--optimize_camera_registration True --load_registration True --orientation-method none --center-method none --auto-scale-poses False"

    print(scene_names)

    exps = []

    for scene in scene_names:
        exp_params = {
            "data1": f"{data_dir}/{scene}/transforms_1.json",
            "data2": f"{data_dir}/{scene}/transforms_2.json",
            "experiment_name": f"{scene}",
            "scene_name": f"{scene}",
            "downscale_factor": f"{int(1)}",
            "reg_downscale_factor": f"{int(1)}",
            "num_points_reg": "10",
            "num_points_unreg": "10",
            "pretrain-iters": "0",
            "unreg_data_dir": f"{data_dir}_unreg/",
            "outputs_dir": f"{outputs_dir}"
        }
        exps.append(exp_params)
    total_stats = {}
    for exp in exps:
        print("experiment_name:", exp["experiment_name"])
        registered_scene_cmd = default_params + "--output-dir " + exp["outputs_dir"] + " --machine.seed {}"\
                               " --data " + exp["data1"] + " --experiment_name " + exp["experiment_name"]  \
                               + "_registered --timestamp " + timestamp + default_params_registered + \
                               "--downscale_factor " + exp["downscale_factor"] + \
                               " --objaverse_transform_matrix " + "0"

        unregistered_scene_cmd = default_params + "--output-dir " + exp["outputs_dir"] + " --machine.seed {}" \
                               " --data " + exp["data2"] + " --experiment_name " + exp["experiment_name"]  \
                               + "_unregistered --timestamp " + timestamp + default_params_unregistered + \
                               "--downscale_factor " + exp["downscale_factor"] + \
                               " --objaverse_transform_matrix " + "1"

        export_cmd_unreg = "ns-export nf-cameras --seed {} --load-config " + outputs_dir + exp["experiment_name"] + "_unregistered/nerfacto/" \
                     + timestamp + "/config.yml" + " --output-dir " + exp["unreg_data_dir"] + exp["experiment_name"] + "_unregistered" \
                     + " --num_points " + exp["num_points_unreg"] + " --downscale_factor " + exp["reg_downscale_factor"] \
                     + " --min_depth " + "4" + " --max_depth " + "4" \
                     + " --near_plane " + "0" + " --far_plane " + "10" \
                     + " --threshold 0.001"

        export_unreg_pcd = "ns-export nf-pointcloud --load-config " + outputs_dir + exp["experiment_name"] + "_unregistered/nerfacto/" \
                     + timestamp + "/config.yml" \
                     + "  --output-dir " + outputs_dir + exp["experiment_name"] + "_unregistered/nerfacto/" + timestamp +"/"\
                     + " --bounding-box-min -1.5 -1.5 -1.5 --bounding-box-max 1.5 1.5 1.5 " \
                     + " --remove_outliers False"

        export_reg_pcd = "ns-export nf-pointcloud --load-config " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" \
                     + timestamp + "/config.yml" \
                     + "  --output-dir " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp +"/"\
                     + " --bounding-box-min -1.5 -1.5 -1.5 --bounding-box-max 1.5 1.5 1.5 " \
                     + " --remove_outliers False"

        fgr_cmd = "python scripts/fgr.py " + outputs_dir + exp["experiment_name"] + "_unregistered/nerfacto/" \
                     + timestamp + "/point_cloud.ply " \
                     + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" \
                     + timestamp + "/point_cloud.ply " \
                     + outputs_dir + exp["experiment_name"] + "_unregistered/nerfacto/" \
                     + timestamp + "/t0.json"

        registeration_cmd = default_params_registration + " --output-dir " + exp["outputs_dir"] + \
                            " --pretrain-iters " + exp["pretrain-iters"] + " --machine.seed {}" \
                            + " --t0 " +outputs_dir + exp["experiment_name"] + "_unregistered/nerfacto/" \
                            + timestamp + "/t0.json" \
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
            os.system(export_unreg_pcd)
            os.system(export_reg_pcd)
            os.system(fgr_cmd)
            os.system(registeration_cmd.format(str(scene_seed*i)))

            # Read the stats of the registration
            exp_stats_path = outputs_dir + exp["experiment_name"] + "_registration/nerfacto/" + timestamp + "/stats.json"
            with open(os.path.join(exp_stats_path), 'r') as f:
                exp_stats = json.load(f)
            stats_list.append(exp_stats)
            if exp_stats["psnr"] > best_psnr:
                best_psnr = exp_stats["psnr"]
                best_exp_stats = exp_stats
                print("rotation_rmse", best_exp_stats["rotation_rmse"])
                print("translation_rmse_100", best_exp_stats["translation_rmse_100"])

        total_stats[exp["experiment_name"]] = {"best": best_exp_stats, "stats_list": stats_list}

    curr_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    base_dir = f"{outputs_dir}/../stats/"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    total_stats_path = base_dir + curr_timestamp + str(random.randint(0, 100)) + ".json"
    with open(total_stats_path, "w") as outfile:
        json.dump(total_stats, outfile, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Usage: python reg_pipeline.py <data_directory> <output_directory> <scene_names> <exp_types> <downscale> <<timestamp>>")
    else:
        base_directory = sys.argv[1]
        output_directory = sys.argv[2]
        if len(sys.argv) >=4:
            scene_names = sys.argv[3].split(',')
        else:
            scene_names = None

        if scene_names[0] == "all":
            scene_names = None

        if not os.path.isdir(base_directory):
            print(f"Error: {base_directory} is not a valid directory.")
        elif not os.path.isdir(output_directory):
            print(f"Error: {output_directory} is not a valid directory.")
        else:
            if len(sys.argv) == 5:
                timestamp = sys.argv[4]
                main(base_directory, output_directory, scene_names, timestamp)
            else:
                main(base_directory, output_directory, scene_names)




