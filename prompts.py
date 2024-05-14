models = ["MV"]
loss_fn1 = ["VSD"]
loss_fn2 = ["MSE"]
prompts = [
    # ("a chair", "a chair"),
    ("a photo of a teapot", "a teapot"),
    ("a high-quality picture of a horse", "a horse"),
    # ("a pair of sunglasses", "a pair of sunglasses"),
    ("A DSLR photo of a fast red sportscar", "a fast sportscar"),
    # ("Two cars racing on a race track", "a racetrack"),
    # ("An 8K picture of a delicious ice corn", "An ice corn"),
    # ("A shark jumping", "a shark"),
    # ("A shark attacking a fishers boat","A shark attacking a fishers boat"),
    # ("A DSLR photo of a monkey and a dinosaur playing chess", "a monkey and a dinosaur playing chess"),
    # ("A wooden bench in a park full of trees", "a park"),
    # ("A man fishing in the pond with a large fountain", "a fountain"),
]

count = 0
for prompt, shap_txt in prompts:
    for model in models:
        for stage1 in loss_fn1:
            for stage2 in loss_fn2:
                for shape in [False]:
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
                        cmd1 += f' point_e="SHAPE_{shap_txt}" num_pts=20'
                        cmd2 += f' point_e="SHAPE_{shap_txt}" num_pts=20'
                    cmd1 += f' stage1="{stage1}" stage2="{stage2}"'
                    cmd2 += f' stage1="{stage1}" stage2="{stage2}"'
                    cmd1 += f" outdir=logs/{model}_st1_{stage1}_st2_{stage2}_shape_{shape} save_path={prompt.replace(' ', '_')}"
                    cmd2 += f" outdir=logs/{model}_st1_{stage1}_st2_{stage2}_shape_{shape} save_path={prompt.replace(' ', '_')}"
                    count += 1
                    print(cmd1)
                    print(cmd2)
                    print()
print(count)
