import os
from datetime import datetime
from pprint import pprint

import yaml

from lib.txt2img import Txt2Img


def main():
    # now = datetime.now().strftime("%m%d")
    now = datetime.now().strftime("%m%d")
    # output_dir = f"./img/{now}/18332684952414045286"
    output_dir = f"./img/{now}"
    os.makedirs(output_dir, exist_ok=True)

    # prompt = "masterpiece, best quality, detailed Face, 1girl, small breasts, solo, Sexy Cleavage, ao dai, bangs, messy hair, hair between eyes, sexy face, (from below), (Perfectly Drawn Hands), photorealistic, ultra detailed, (cowboy shot), <lora:raidenShogunRealistic_raidenshogunHandsfix:0.7>"
    # negative_prompt = "(worst quality, low quality:1.3), (depth of field, blurry:1.2), (greyscale, monochrome:1.1), 3D face, nose, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title, (tan, muscular, loli, petite, child, infant, toddlers, chibi, sd character:1.1), multiple view, Reference sheet"

    # agent = Txt2Img(output_dir=output_dir, width=480, height=640)
    # agent.gen(prompt, negative=negative_prompt, n=20)

    # ToDo: プロンプトとネガティブプロンプト or プロンプトファイル、出力ディレクトリの指定、生成回数をCLIから読み込めるようにする
    # ToDo: デバッグフラグをCLIから読み込み.デバッグ時は画像、プロンプト、設定値を1つのフォルダに格納する
    #       ディレクトリ名は月日時分(%m%d%H%M)、画像名は[seed]_000.png、設定値ファイル名はconfig.yaml

    yaml_file = os.path.join(os.getcwd(), "prompts/prompt_dev.yaml")
    with open(yaml_file, "r") as yml:
        config = yaml.safe_load(yml)
    prompt = ",".join(config["prompt"])
    negative_prompt = ",".join(config["negative_prompt"])

    # agent = Txt2Img(output_dir=output_dir, model_id="stabilityai/stable-diffusion-2-1")
    agent = Txt2Img(output_dir=output_dir, width=336, height=448)
    images = agent.gen(prompt, negative=negative_prompt, n=1)

    print("Output files:")
    pprint(images)

    # agent = Txt2Img(output_dir=output_dir, seed=18332684952414045286, width=384, height=512)

    # # pose
    # types = ["", "restrained", "symmetry", "pov", "foot focus", "ass focus", "upskirt", "head_out_of_frame", "profile"]

    # # camera angle
    # types = types + ["from below", "from above", "from behind", "from side", "looking at viewer", "looking away", "looking back", "looking down", "looking up", "looking to the side", "looking afar"]

    # # zoom size
    # types = types + ["face", "close up", "portrait", "upper body", "cowboy shot", "full body", "lower body"]
    # for i, v in enumerate(types):
    #     i = i + 8
    #     v = v.replace(" ", "-")
    #     p = "{{%s}},%s" % (prompt, v)
    #     images = agent.gen(p, negative=negative_prompt, n=1, file_name=f"{str(i).zfill(3)}_{v}")

    #     print("Output files:")
    #     pprint(images)

if __name__ == "__main__":
    main()
