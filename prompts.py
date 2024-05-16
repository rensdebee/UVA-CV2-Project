models = ["MV"]
loss_fn1 = ["ISM"]
loss_fn2 = ["MSE"]
shap = [False]

prompts = [
    ("a high-quality photo realistic render of a chair", "a chair"),
    ("a 4k photo of a white teapot", "a teapot"),
    ("a high-quality 4k picture of a brown horse", "a horse"),
    ("a pair of sunglasses", "a pair of sunglasses"),
    ("A DSLR photo of a fast red sportscar", "a sportscar"),
    (
        "An 8K picture of a delicious ice corn with vanilla and chocolade ice",
        "An ice corn",
    ),
    ("A realistic shark jumping out of the water", "a shark"),
    (
        "A shark attacking a fisher's boat in 4k with high realism",
        "A shark attacking a fisher's boat",
    ),
    ("An open laptop playing a video of a dancing man", "An open laptop"),
    ("A true to life 8k corgi dog", "A dog"),
]

count = 0
for prompt, shap_txt in prompts:
    for model in models:
        for stage1 in loss_fn1:
            for stage2 in loss_fn2:
                for shape in shap:
                    cmd1 = (
                        f'python main.py --config configs/text.yaml prompt="{prompt}"'
                    )
                    cmd2 = (
                        f'python main2.py --config configs/text.yaml prompt="{prompt}"'
                    )
                    if model == "MV":
                        cmd1 += f" mvdream=True"
                        cmd2 += f" mvdream=True"
                    if shape:
                        cmd1 += f' point_e="SHAPE_{shap_txt}"'
                        cmd2 += f' point_e="SHAPE_{shap_txt}"'
                    cmd1 += f' stage1="{stage1}" stage2="{stage2}"'
                    cmd2 += f' stage1="{stage1}" stage2="{stage2}"'
                    cmd1 += f" outdir=logs/{model}_st1_{stage1}_st2_{stage2}_shape_{shape} save_path={prompt.replace(' ', '_')}"
                    cmd2 += f" outdir=logs/{model}_st1_{stage1}_st2_{stage2}_shape_{shape} save_path={prompt.replace(' ', '_')}"
                    count += 1
                    print(cmd1)
                    print(cmd2)
                    print()
print(count)
