import os
import torch
import time
import pprint
from diffusers import StableDiffusionPipeline
from datetime import datetime
from diffusers.pipelines.stable_diffusion import safety_checker


class Txt2Img:
    """AIを利用してテキストから画像を生成"""

    def __init__(
        self,
        model_id: str = "hakurei/waifu-diffusion",
        width: int = 400,
        height: int = 400,
        seed: int = 0,
        steps: int = 28,
        revision: str = "fp16",
        low_cpu_mem_usage: bool = True,
        device: str = "cpu",
        scale: float = 11.0,
        output_dir: str = "./",
    ) -> None:
        """_summary_

        Args:
            model_id (str, optional): AIモデル. Defaults to "hakurei/waifu-diffusion".
            width (int, optional): 横幅(px). Defaults to 400.
            height (int, optional): 縦幅(px). Defaults to 400.
            seed (int, optional): 乱数生成シード.デフォルトの場合、乱数を自動生成. Defaults to 0.
            steps (int, optional): ステップ数.大きいほど書き込みが多くなる. Defaults to 28.
            revision (str, optional): ブランチ名. Defaults to "fp16".
            low_cpu_mem_usage (bool, optional): 省メモリ化. Defaults to True.
            device (str, optional): 生成に使用するデバイス. Defaults to "cpu".
            guidance_scale (float, optional): プロンプトへの従属度.大きすぎると破綻する. Defaults to 11.0.
            output_dir (str, optional): 出力ディレクトリ名. Defaults to "./".
        """
        self.model_id = model_id
        self.width = width
        self.height = height
        self.seed = seed
        self.steps = steps
        self.revision = revision
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.device = device
        self.scale = scale
        self.output_dir = output_dir

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            revision=self.revision,
            # custom_pipeline="lpw_stable_diffusion",  # プロンプトの長文対応
        ).to(self.device)

    def sc(self, clip_input, images):
        return images, [False for i in images]

    safety_checker.StableDiffusionSafetyChecker.forward = sc

    def gen(
        self,
        prompt: str,
        negative: str = "",
        n: int = 1,
    ) -> None:
        """指定回数

        Args:
            prompt (str): 画像生成テキスト
            negative (str, optional): ネガティブプロンプト. Defaults to "".
            n (int, optional): 生成回数. Defaults to 1
        """

        outputs: list[str] = []
        num = 1
        while True:
            file_path = self.__exec(prompt, negative)
            outputs.append(file_path)
            num += 1

            if num > n:
                break
            else:
                time.sleep(3)

        print("Output files:")
        pprint.pprint(outputs)

    def __exec(
        self,
        prompt: str,
        negative: str = "",
    ) -> str:
        """テキストから画像生成

        Args:
            prompt (str): 画像生成テキスト
            negative (str, optional): ネガティブプロンプト. Defaults to "".

        Returns:
            str: 出力ファイル名
        """

        generator = torch.Generator(device=self.device)
        img_seed = self.seed
        if self.seed is 0:
            img_seed = generator.seed()

        generator = generator.manual_seed(img_seed)

        config = {
            "prompt": prompt,
            "negative_prompt": negative,
            "seed": img_seed,
            "scale": self.scale,
            "steps": self.steps,
        }
        pprint.pprint(config)

        image = self.pipe(
            prompt,
            negative_prompt=negative,
            num_inference_steps=self.steps,
            width=self.width,
            height=self.height,
            generator=generator,
            guidance_scale=self.scale,
            # max_embeddings_multiples=2,  # プロンプトの長文対応
        ).images[0]

        try:
            now = datetime.now().strftime("%m%d%H%M")
            file_path = os.path.join(self.output_dir, f"{now}_{img_seed}.png")
            image.save(file_path)
            return file_path
        except:
            image.save("output.png")
            return "output.png"
