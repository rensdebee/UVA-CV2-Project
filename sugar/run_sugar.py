import os
import glob
import argparse
import numpy as np

def find_latest_path(paths):
    # print(paths)
    nums = [int(n.split("@")[-1].replace("-", "")) for n in paths]
    idx = np.argmax(nums)
    latest_path = paths[idx]
    return latest_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--asset_name", type=str, default=None,
                        help="The name of the asset to be generated")
    parser.add_argument("-c", "--condition_type", type=str, default="depth",
                        help="The type of input condition, should be one of [canny, depth, normal, scribble]")
    parser.add_argument("-p", "--prompt", type=str, default=None,
                        help="Prompt of the asset")
    
    parser.add_argument('--gs_path', type=str, default=None, help='Input file')
    parser.add_argument('--resume', default=False, action=argparse.BooleanOptionalAction, help='Resume the mesh project')
    
    # Gaussian refinement config
    parser.add_argument("--load_coarse_gs_only_position", action="store_true", default=False,
                        help="Use only coarse gaussian's position for initialization when gaussian refinement.")
    
    # SuGaR extraction config
    parser.add_argument("--surface_level", type=float, default=0.3,
                        help="Surface level for coarse sugar extraction")
    parser.add_argument("--decimation_target", type=int, default=200_000,
                        help="Decimation target for coarse sugar extraction")
    parser.add_argument("--poisson_depth", type=int, default=6,
                        help="Poisson depth for coarse sugar extraction")
    
    
    args = parser.parse_args()

    # Setup
    asset_name = args.asset_name
    condition_type = args.condition_type
    prompt = args.prompt

    exp_root_dir = f"workspace/mvcontrol_{condition_type}"

    if not args.gs_path:
        args.gs_path = args.prompt.lower().replace(" ", "_") + ".ply"

    for f in glob("sugar/logs/*/*.ply", recursive=True):
        if "args.gs_path" in f:
            coarse_gs_path = f
    
    assert os.path.exists(coarse_gs_path), f"The coarse gaussian of path '{coarse_gs_path}' do not exist!"
    
    os.chdir('sugar')
    
    directory = os.path.join(exp_root_dir, asset_name)
    print(directory)
    if args.resume and os.path.isdir(directory):
        print("Resuming from stage 2 refined gaussians")
        pass
    else:
        # Stage 2. Gaussian refinement with SDS (MVControl + DeepFloyd-IF)
        print('Starting stage 2, gaussian refinement')
        command = f"python launch.py --config custom/threestudio-3dgs/configs/mvcontrol-gaussian.yaml --train --gpu 0 \
                system.stage=gaussian \
                system.control_condition_type={condition_type} \
                system.prompt_processor.prompt='{prompt}' \
                system.geometry.geometry_convert_from={coarse_gs_path} \
                system.guidance_control.pretrained_controlnet_name_or_path=lzq49/mvcontrol-4v-depth \
                exp_root_dir={exp_root_dir} name={asset_name} tag=gaussian_refine"
                
        if args.load_coarse_gs_only_position:
            command += " system.geometry.load_ply_only_vertex=true system.geometry.load_vertex_only_position=true"
        os.system(command)

    # Stage 3. Coarse SuGaR extraction from refined gaussians
    refined_gs_trial_dir = find_latest_path(
        glob.glob(
            os.path.join(exp_root_dir, asset_name, "gaussian_refine*")
        )
    )
    refined_gs_path = glob.glob(
        os.path.join(refined_gs_trial_dir, "save", "exported_gs_step*.ply")
    )[0]
    # print(os.path.join(exp_root_dir, asset_name, "gaussian_refine*"))
    print(refined_gs_path)
    coarse_sugar_output_dir = os.path.join(
        exp_root_dir, asset_name, "coarse_sugar"
    )
    # print(coarse_gs_path)
    command = f"python extern/sugar/extract_mesh.py -s extern/sugar/load/scene \
            -c {refined_gs_path} -l {args.surface_level} -d {args.decimation_target} \
            -o {coarse_sugar_output_dir} \
            --poisson_depth {args.poisson_depth} \
            --use_vanilla_3dgs"
    os.system(command)
    
    # Stage 4. SuGaR refinement (VSD)
    sugar_mesh_path = os.path.join(
        coarse_sugar_output_dir, f"sugarmesh_vanilla3dgs_level{args.surface_level}_decim{args.decimation_target}_pd{args.poisson_depth}.ply" 
    )
    command = f"python launch.py --config custom/threestudio-3dgs/configs/mvcontrol-sugar-vsd.yaml --train --gpu 0 \
        system.stage=sugar \
        system.control_condition_type={condition_type} \
        system.geometry.surface_mesh_to_bind_path={sugar_mesh_path} \
        system.prompt_processor.prompt='{prompt}' \
        system.guidance_control.pretrained_controlnet_name_or_path=lzq49/mvcontrol-4v-depth\
        exp_root_dir={exp_root_dir} \
        name={asset_name} \
        tag=sugar_refine"
    os.system(command)

    # Stage 5. Texture mesh extraction
    sugar_out_dir = find_latest_path(
        glob.glob(
            os.path.join(exp_root_dir, asset_name, "sugar_refine*")
        )
    )
    command = f"python launch.py --config {sugar_out_dir}/configs/parsed.yaml --export --gpu 0 \
        resume={sugar_out_dir}/ckpts/last.ckpt"
    os.system(command)




    