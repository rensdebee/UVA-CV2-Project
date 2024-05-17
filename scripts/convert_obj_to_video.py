import os
import glob
import argparse
import numpy as np
from PIL import Image as im

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir", default="./logs", type=str, help="Directory where obj files are stored"
)
parser.add_argument(
    "--out", default="videos", type=str, help="Directory where videos will be saved"
)
args = parser.parse_args()

grey = np.ones((1024, 1024), dtype=np.uint8) * 148
# Make 3-channel from singe-channel
grey_image = np.dstack((grey, grey, grey))
grey_image = im.fromarray(grey_image)

out = args.out
os.makedirs(out, exist_ok=True)

files = glob.glob(f"{args.dir}/**/*.obj")
for f in files:
    name = os.path.basename(f)
    # first stage model, ignore
    if name.endswith("_mesh.obj"):
        continue
    # print(f'[INFO] process {name}')

    # MP4 with texture
    texture_out_path = os.path.join(
        args.out, f.replace("./", "").replace(name, "texture")
    )
    out_path = os.path.join(texture_out_path, name.replace(".obj", ".mp4"))
    os.makedirs(texture_out_path, exist_ok=True)
    os.system(f"python -m kiui.render {f} --save_video {out_path} --wogui")

    no_texture_out_path = os.path.join(
        args.out, f.replace("./", "").replace(name, "shape")
    )
    out_path = os.path.join(no_texture_out_path, name.replace(".obj", ".mp4"))
    os.makedirs(no_texture_out_path, exist_ok=True)
    og_mesh_path = f.replace(".obj", "_albedo.png")
    os.rename(og_mesh_path, og_mesh_path.replace(".png", ".#png"))
    grey_image.save(og_mesh_path)
    os.system(
        f"python -m kiui.render {f} --save_video {out_path} --wogui --mode lambertian"
    )
    os.remove(og_mesh_path)
    mesh_path = og_mesh_path.replace(".png", ".#png")
    os.rename(mesh_path, mesh_path.replace(".#png", ".png"))
