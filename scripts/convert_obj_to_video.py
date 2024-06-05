import os
import glob
import argparse
import numpy as np
from PIL import Image as im
import torch
import numpy as np
from torchvision import transforms as T
from transformers import CLIPModel, CLIPProcessor
import tqdm
import json
from kiui.render import GUI
import dearpygui.dearpygui as dpg


# eval the clip-similarity for an input image and a geneated mesh
class CLIP:
    def __init__(self, device, model_name="openai/clip-vit-large-patch14"):

        self.device = device

        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, image):
        # image: PIL, np.ndarray uint8 [H, W, 3]

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )  # normalize features

        return image_features

    def encode_text(self, text):
        # text: str

        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        text_features = self.clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )  # normalize features

        return text_features


def clip_sim(
    prompt,
    mesh,
    clip,
    opt,
    front_dir="+z",
):

    opt.wogui = False
    opt.prompt = prompt
    opt.mesh = mesh
    opt.front_dir = front_dir
    gui = GUI(opt)
    with torch.no_grad():
        ref_features = clip.encode_text(opt.prompt)

    # render from random views and evaluate similarity
    results = []

    elevation = [
        opt.elevation,
    ]
    azimuth = np.linspace(0, 360, opt.num_azimuth, dtype=np.int32, endpoint=False)
    for ele in tqdm.tqdm(elevation):
        for azi in tqdm.tqdm(azimuth):
            gui.cam.from_angle(ele, azi)
            gui.need_update = True
            gui.step()
            image = (gui.render_buffer * 255).astype(np.uint8)
            with torch.no_grad():
                cur_features = clip.encode_image(image)

            similarity = (ref_features * cur_features).sum(dim=-1).mean().item()

            results.append(similarity)
    dpg.destroy_context()
    avg_similarity = np.mean(results)
    return avg_similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbr", action="store_true", help="enable PBR material")
    parser.add_argument(
        "--envmap", type=str, default=None, help="hdr env map path for pbr"
    )
    parser.add_argument(
        "--mode",
        default="albedo",
        type=str,
        choices=["lambertian", "albedo", "normal", "depth", "pbr"],
        help="rendering mode",
    )
    parser.add_argument("--W", type=int, default=800, help="GUI width")
    parser.add_argument("--H", type=int, default=800, help="GUI height")
    parser.add_argument(
        "--ssaa", type=float, default=1, help="super-sampling anti-aliasing ratio"
    )
    parser.add_argument(
        "--radius", type=float, default=3, help="default GUI camera radius from center"
    )
    parser.add_argument(
        "--fovy", type=float, default=50, help="default GUI camera fovy"
    )
    parser.add_argument(
        "--force_cuda_rast",
        action="store_true",
        help="force to use RasterizeCudaContext.",
    )
    parser.add_argument("--elevation", type=int, default=0, help="rendering elevation")
    parser.add_argument(
        "--num_azimuth",
        type=int,
        default=8,
        help="number of images to render from different azimuths",
    )
    parser.add_argument(
        "--dir", default="./logs", type=str, help="Directory where obj files are stored"
    )
    parser.add_argument(
        "--out", default="videos", type=str, help="Directory where videos will be saved"
    )

    parser.add_argument(
        "--clip",
        action="store_true",
        help="Bool indicating if clip score needs to be calculated",
    )

    args = parser.parse_args()

    grey = np.ones((1024, 1024), dtype=np.uint8) * 148
    # Make 3-channel from singe-channel
    grey_image = np.dstack((grey, grey, grey))
    grey_image = im.fromarray(grey_image)

    out = args.out
    os.makedirs(out, exist_ok=True)

    if args.clip:
        clip_scores = {}
        clip = CLIP("cuda", model_name="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    clip_scores = {}

    files = glob.glob(f"{args.dir}/**/*.obj")
    for f in files:
        name = os.path.basename(f)
        # first stage model, ignore
        if name.endswith("_mesh.obj"):
            continue
        prompt = name.replace("_", " ").replace(".obj", "")
        config = f.split("\\")[-2]

        if "sugar" in config.lower():
            sugar = True
        else:
            sugar = False

        if args.clip:
            if config not in clip_scores.keys():
                clip_scores[config] = {}
            if prompt not in clip_scores[config]:
                clip_scores[config][prompt] = {}

            clip_scores[config][prompt]["shape"] = []
            clip_scores[config][prompt]["texture"] = []

        print(f"[INFO] processing {config} --- {prompt}")

        # texture
        texture_out_path = os.path.join(
            args.out, f.replace("./", "").replace(name, "texture")
        )
        out_path = os.path.join(texture_out_path, name.replace(".obj", ".mp4"))
        os.makedirs(texture_out_path, exist_ok=True)
        if sugar:
            os.system(
                f"python -m kiui.render {f} --save_video {out_path} --radius {args.radius} --wogui --front_dir '+y'"
            )
            if args.clip:
                clip_text = clip_sim(prompt, f, clip, args, "+y")
                clip_scores[config][prompt]["texture"].append(clip_text)
        else:
            os.system(
                f"python -m kiui.render {f} --save_video {out_path} --radius {args.radius} --wogui"
            )
            if args.clip:
                clip_text = clip_sim(prompt, f, clip, args)
                clip_scores[config][prompt]["texture"].append(clip_text)
        # no texture
        no_texture_out_path = os.path.join(
            args.out, f.replace("./", "").replace(name, "shape")
        )
        out_path = os.path.join(no_texture_out_path, name.replace(".obj", ".mp4"))
        os.makedirs(no_texture_out_path, exist_ok=True)
        if sugar:
            og_mesh_path = f.replace(".obj", ".png")
        else:
            og_mesh_path = f.replace(".obj", "_albedo.png")
        os.rename(og_mesh_path, og_mesh_path.replace(".png", ".#png"))
        grey_image.save(og_mesh_path)

        if sugar:
            os.system(
                f"python -m kiui.render {f} --save_video {out_path} --radius {args.radius} --wogui --front_dir '+y'"
            )
            if args.clip:
                clip_shape = clip_sim(prompt, f, clip, args, "+y")
                clip_scores[config][prompt]["shape"].append(clip_shape)
        else:
            os.system(
                f"python -m kiui.render {f} --save_video {out_path} --radius {args.radius} --wogui"
            )
            if args.clip:
                clip_shape = clip_sim(prompt, f, clip, args)
                clip_scores[config][prompt]["shape"].append(clip_shape)

        os.remove(og_mesh_path)
        mesh_path = og_mesh_path.replace(".png", ".#png")
        os.rename(mesh_path, mesh_path.replace(".#png", ".png"))

    for config in clip_scores:
        shape_scores = []
        text_score = []
        for prompt in clip_scores[config]:
            shape_scores.append(clip_scores[config][prompt]["shape"])
            text_score.append(clip_scores[config][prompt]["texture"])
        print(
            f"Config: {config} \n shape: {np.mean(shape_scores)} \n texture: {np.mean(text_score)} \n"
        )
    # Convert and write clip scores to file
    with open(args.out + "/clip.json", "w") as outfile:
        json.dump(clip_scores, outfile)
