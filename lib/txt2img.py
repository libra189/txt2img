import os
import time
from datetime import datetime

import torch
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import safety_checker


class Txt2Img:
    """AIを利用してテキストから画像を生成"""

    def __init__(
        self,
        model_id: str = "hakurei/waifu-diffusion",
        vae: str = "stabilityai/sd-vae-ft-mse",
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
        """AIモデル、処理設定、生成する画像の設定

        Args:
            model_id (str, optional): AIモデル. Defaults to "hakurei/waifu-diffusion".
            vae (str, optional): 画質を向上するための改良オートエンコーダー. Defaults to "stabilityai/sd-vae-ft-mse".
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
        # ToDo: 引数のバリデーション

        self.model_id = model_id
        self.vae = vae
        self.width = width
        self.height = height
        self.seed = seed
        self.steps = steps
        self.revision = revision
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.device = device
        self.scale = scale
        self.output_dir = output_dir

        vae = AutoencoderKL.from_pretrained(self.vae)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            revision=self.revision,
            vae=vae, # FixMe: 警告が多いため一時的にコメントアウト
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
        file_name: str = "",
    ) -> list[str]:
        """指定回数

        Args:
            prompt (str): 画像生成テキスト
            negative (str, optional): ネガティブプロンプト. Defaults to "".
            n (int, optional): 生成回数. Defaults to 1
            file_name (str, optional): 出力画像のファイル名. Defaults to "".
        """

        outputs: list[str] = []
        num = 1
        while True:
            img_name = file_name
            if (len(img_name) > 0 and n > 1):
                img_name = f"{str(num).zfill(3)}_{img_name}"

            file_path = self.__exec(prompt, negative, img_name)
            outputs.append(file_path)
            num += 1

            if num > n:
                break
            else:
                time.sleep(3)

        return outputs

    def __exec(
        self,
        prompt: str,
        negative: str,
        img_name: str
    ) -> str:
        """テキストから画像生成

        Args:
            prompt (str): 画像生成テキスト.
            negative (str, optional): ネガティブプロンプト.
            img_name (str, optional): 出力画像のファイル名.

        Returns:
            str: 出力ファイル名
        """

        generator = torch.Generator(device=self.device)
        img_seed = self.seed
        if self.seed == 0:
            img_seed = generator.seed()

        generator = generator.manual_seed(img_seed)

        # ToDo： 設定出力はデバックモードのみ
        # config = {
        #     "prompt": prompt,
        #     "negative_prompt": negative,
        #     "seed": img_seed,
        #     "scale": self.scale,
        #     "steps": self.steps,
        # }
        # pprint.pprint(config)

        image = self.pipe(
            prompt,
            negative_prompt=negative,
            num_inference_steps=self.steps,
            width=self.width,
            height=self.height,
            generator=generator,
            guidance_scale=self.scale,
            # max_embeddings_multiples=10,  # プロンプトの長文対応
        ).images[0]

        try:
            if len(img_name) > 0:
                file_path = os.path.join(self.output_dir, f"{img_name}.png")
            else:
                now = datetime.now().strftime("%m%d%H%M")
                file_path = os.path.join(self.output_dir, f"{now}_{img_seed}.png")

            image.save(file_path)
            return file_path
        except:
            image.save("output.png")
            return "output.png"
