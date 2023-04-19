import os
from datetime import datetime
from pprint import pprint

import yaml

from lib.txt2img import Txt2Img


def main():
    now = datetime.now().strftime("%m%d")
    os.makedirs(f"./img/{now}", exist_ok=True)
    output_dir = f"./img/{now}"

    # prompt = "masterpiece, best quality, detailed Face, 1girl, small breasts, solo, Sexy Cleavage, ao dai, bangs, messy hair, hair between eyes, sexy face, (from below), (Perfectly Drawn Hands), photorealistic, ultra detailed, (cowboy shot), <lora:raidenShogunRealistic_raidenshogunHandsfix:0.7>"
    # negative_prompt = "(worst quality, low quality:1.3), (depth of field, blurry:1.2), (greyscale, monochrome:1.1), 3D face, nose, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title, (tan, muscular, loli, petite, child, infant, toddlers, chibi, sd character:1.1), multiple view, Reference sheet"

    # agent = Txt2Img(output_dir=output_dir, width=480, height=640)
    # agent.gen(prompt, negative=negative_prompt, n=20)

    yaml_file = os.path.join(os.getcwd(), "prompts/prompt.yaml")
    with open(yaml_file, "r") as yml:
        config = yaml.safe_load(yml)
    prompt = ",".join(config["prompt"])
    negative_prompt = ",".join(config["negative_prompt"])

    # ToDo: プロンプトのトークン長チェック.78以上の場合は長文対応処理判定を追加する

    agent = Txt2Img(output_dir=output_dir)
    files = agent.gen(prompt, negative=negative_prompt, n=1)

    print("Output files:")
    pprint(files)

if __name__ == "__main__":
    main()
