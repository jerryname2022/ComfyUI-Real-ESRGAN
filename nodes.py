import os
from pathlib import Path
import torch
import numpy as np
import random
import time
import cv2
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import torchvision.transforms.functional as TF
from torchvision.ops import masks_to_boxes

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

import folder_paths
from PIL import Image, ImageFilter
from .gfpgan.realesrgan.basicsr.rrdbnet_arch import RRDBNet
from .gfpgan.realesrgan.basicsr.utils import img2tensor, tensor2img

from .gfpgan.gfpgan_bilinear_arch import GFPGANBilinear
from .gfpgan.gfpganv1_arch import GFPGANv1
from .gfpgan.gfpganv1_clean_arch import GFPGANv1Clean


class RealESRGANModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B", "RealESRGAN_x2plus"],
                    {"default": "RealESRGAN_x4plus"}),
                "half": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("RealESRGAN_Model",)
    FUNCTION = "load_model"

    CATEGORY = "Real-ESRGAN/Model"

    def load_model(self, model_name, half):

        model_path = os.path.join(os.path.join(folder_paths.models_dir, "Real-ESRGAN"), "{}.pth".format(model_name))
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # prefer to use params_ema
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'

        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        else:  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

        # net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model.load_state_dict(state_dict[keyname], strict=True)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        if half:
            model = model.half()

        return (model,)


class GFPGANModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["GFPGANv1.3", "GFPGANv1.4"],
                    {"default": "GFPGANv1.4"}),
                "arch": (
                    ["clean", "bilinear", "original", "RestoreFormer"],
                    {"default": "clean"}),
            }
        }

    RETURN_TYPES = ("GFPGAN_Model",)
    FUNCTION = "load_model"

    CATEGORY = "Real-ESRGAN/Model"

    def load_model(self, model_name, arch):

        upscale = 4
        channel_multiplier = 2
        model_root = os.path.join(os.path.join(folder_paths.models_dir, "Real-ESRGAN"))
        model_path = os.path.join(model_root, "{}.pth".format(model_name))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # initialize the GFP-GAN
        if arch == 'clean':
            gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'bilinear':
            gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'original':
            gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'RestoreFormer':
            from gfpgan.restoreformer_arch import RestoreFormer
            gfpgan = RestoreFormer()

        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'

        gfpgan.load_state_dict(loadnet[keyname], strict=True)
        gfpgan.eval()
        gfpgan = gfpgan.to(device)

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=device,
            model_rootpath=model_root)
        result = {"gfpgan": gfpgan, "face_helper": face_helper}

        return (result,)


class InpaintingLamaModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["iic/cv_fft_inpainting_lama"],
                    {"default": "iic/cv_fft_inpainting_lama"}),
            }
        }

    RETURN_TYPES = ("InpaintingLama_Model",)
    FUNCTION = "load_model"

    CATEGORY = "Real-ESRGAN/Model"

    @torch.no_grad()
    def load_model(self, model_name):
        model_path = os.path.join(os.path.join(folder_paths.models_dir, model_name))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        inpainting = pipeline(Tasks.image_inpainting, model=model_path, refine=False, device=device)

        return (inpainting,)


class InpaintingLamaImageGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_model": ("InpaintingLama_Model",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.25, "min": 0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "gan_image"
    CATEGORY = "Real-ESRGAN/Mask"

    @torch.no_grad()
    def gan_image(self, image_model, image, mask, threshold=0.25):
        print(image.shape, mask.shape)

        rgb_pil = Image.fromarray(
            torch.clamp(torch.round(255.0 * image[0]), 0, 255)
            .type(torch.uint8)
            .cpu()
            .numpy()
        ).convert("RGB")

        alpha_pil = Image.fromarray(
            torch.clamp(torch.round(255.0 * mask[0]), 0, 255)
            .type(torch.uint8)
            .cpu()
            .numpy()
        ).convert("L")

        # print(image.shape, mask.shape, np.array(rgb_pil).shape, np.array(alpha_pil).shape)

        input = {
            'img': rgb_pil,
            'mask': alpha_pil,
        }

        results = image_model(input)
        result = results[OutputKeys.OUTPUT_IMG]

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        image_rgb = torch.from_numpy(result_rgb) / 255
        image_rgb = image_rgb.unsqueeze(0)

        torch.cuda.empty_cache()

        return (image_rgb, mask)


class ImageMergeGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "background_image": ("IMAGE",),
                "blend_width": ("INT", {"default": 5, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "gan_image"
    CATEGORY = "Real-ESRGAN/Mask"

    def calculate_edge_weight(self, i, j, height, width, blend_width):
        """
        计算边缘融合权重
        """
        # 计算到各边的距离
        dist_top = i
        dist_bottom = height - 1 - i
        dist_left = j
        dist_right = width - 1 - j

        # 找到最近边缘的距离
        min_dist = min(dist_top, dist_bottom, dist_left, dist_right)

        # 计算权重（边缘处权重低，中心处权重高）
        if min_dist < blend_width:
            return min_dist / blend_width
        return 1.0

    @torch.no_grad()
    def gan_image(self, image, background_image, blend_width=5):

        image_width, image_height = image.shape[1], image.shape[2]
        background_width, background_height = background_image.shape[1], background_image.shape[2]

        image_pil = Image.fromarray(
            torch.clamp(torch.round(255.0 * image[0]), 0, 255)
            .type(torch.uint8)
            .cpu()
            .numpy()
        ).convert("RGBA")

        background_pil = Image.fromarray(
            torch.clamp(torch.round(255.0 * background_image[0]), 0, 255)
            .type(torch.uint8)
            .cpu()
            .numpy()
        ).convert("RGBA")

        offset_x = (background_width - image_width) // 2
        offset_y = (background_height - image_height) // 2
        if offset_x < 0:
            offset_x = 0
        if offset_y < 0:
            offset_y = 0

        # 转换为numpy数组进行处理
        bg_arr = np.array(background_pil)
        fg_arr = np.array(image_pil)

        fg_height, fg_width = fg_arr.shape[:2]
        # 确保位置在范围内
        offset_x = max(0, min(offset_x, background_width - fg_width))
        offset_y = max(0, min(offset_y, background_height - fg_height))

        # 创建融合区域
        blended = bg_arr.copy()
        roi = blended[offset_y:offset_y + fg_height, offset_x:offset_x + fg_width]

        # 对每个像素进行alpha混合
        for i in range(fg_height):
            for j in range(fg_width):
                fg_pixel = fg_arr[i, j]
                fg_alpha = fg_pixel[3] / 255.0

                if fg_alpha > 0:
                    # 计算边缘权重（距离边缘越近，融合程度越高）
                    edge_weight = self.calculate_edge_weight(i, j, fg_height, fg_width, blend_width)

                    # 混合颜色
                    for c in range(3):
                        roi[i, j, c] = int(
                            fg_pixel[c] * fg_alpha * edge_weight +
                            roi[i, j, c] * (1 - fg_alpha * edge_weight)
                        )

                    # 混合alpha
                    roi[i, j, 3] = max(roi[i, j, 3], int(fg_pixel[3] * edge_weight))

        image_rgb = torch.from_numpy(blended) / 255
        image_rgb = image_rgb.unsqueeze(0)

        torch.cuda.empty_cache()
        return (image_rgb,)


class RealESRGANImageGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_model": ("RealESRGAN_Model",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.25, "min": 0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "gan_image"
    CATEGORY = "Real-ESRGAN/Mask"

    @torch.no_grad()
    def gan_image(self, image_model, image, mask, threshold=0.25):
        image_width, image_height = image.shape[1], image.shape[2]
        mask_width, mask_height = mask.shape[1], mask.shape[2]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # torch.Size([1, 350, 566, 3]) torch.Size([1, 350, 566]) cuda:0
        # print("shape f ", image.shape, mask.shape, device)

        if image_width == mask_width and image_height == mask_height:
            # mask = mask.unsqueeze(-1)  # 形状变为 [b, w, h, c]
            # 提取前三个通道,沿通道维度（dim=3）拼接
            image = torch.cat([image[..., :3], mask.unsqueeze(-1)[..., :1]], dim=3)

        # print("shape b ", image.shape, mask.shape, device)

        image_rgb = image[..., :3]
        image_rgb = image_rgb.permute(0, 3, 1, 2)  # (b, h, w, c) --> (b, c, h, w)

        # TODO unsqueeze torch.Size([3, 3308, 3736]) --> torch.Size([1, 3, 3308, 3736])
        image_rgb = image_rgb.float().to(device)
        image_rgb = image_rgb.half()

        rgb_result = image_model(image_rgb)  # target input shape torch.Size([1, 3, 350, 566])

        result = rgb_result.permute(0, 2, 3, 1).cpu()  # TODO (b, c, h, w) --> (b, h, w, c)

        if image.shape[3] == 4:
            image_alpha = image[..., 3:]
            # print("image_alpha", image_alpha.shape)  # torch.Size([1, 350, 566, 1])

            # torch.Size([1, 350, 566, 1]) --> torch.Size([1, 350, 566, 3])
            image_alpha = image_alpha.expand(-1, -1, -1, 3)
            image_alpha = image_alpha.permute(0, 3, 1, 2)  # (b, h, w, c) --> (b, c, h, w)

            image_alpha = image_alpha.float().to(device)
            image_alpha = image_alpha.half()

            alpha_result = image_model(image_alpha)  # target shape torch.Size([1, 3, 350, 566])

            alpha_result = alpha_result.permute(0, 2, 3, 1).cpu()  # TODO (b, c, h, w) --> (b, h, w, c)
            # # # 提取红色通道（索引 0）
            # alpha_result = alpha_result[..., 0:1]  # 保持维度 [1, 350, 566, 1]

            # 定义灰度化权重（RGB 到灰度的标准公式）
            weights = torch.tensor([0.299, 0.587, 0.114], device=alpha_result.device)
            # 计算加权和，并添加通道维度
            alpha_result = torch.sum(alpha_result * weights, dim=-1, keepdim=True)  # torch.Size([b, h, w, 1])

            alpha_result = alpha_result.to(result.device).type_as(result)

            # 计算全局最小值和最大值
            min_val = alpha_result.min()
            max_val = alpha_result.max()

            # 避免除零（如果所有值相同，设为全0.5）
            if max_val - min_val == 0:
                pass

            # 归一化
            alpha_result = (alpha_result - min_val) / (max_val - min_val)
            # 设定阈值（例如 0.5）
            # threshold = 0.15
            # 二值化：大于等于阈值的设为 1，否则设为 0
            alpha_result = (alpha_result <= threshold).float()

            result = torch.cat([result, alpha_result], dim=3)
            mask = alpha_result.squeeze(-1)  # torch.Size([b, h, w, 1]) --> torch.Size([b, h, w])

        torch.cuda.empty_cache()
        # print("return final", mask.shape, result.shape)
        return (result, mask)


class GFPGANImageGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gfpgan": ("GFPGAN_Model",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "weight": ("FLOAT", {"default": 0.5, "min": 0.01, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "gan_image"
    CATEGORY = "Real-ESRGAN/Mask"

    @torch.no_grad()
    def gan_image(self, gfpgan, image, mask, weight=0.5):

        gfpgan_model = gfpgan["gfpgan"]
        face_helper = gfpgan["face_helper"]

        image_width, image_height = image.shape[1], image.shape[2]
        mask_width, mask_height = mask.shape[1], mask.shape[2]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if image_width == mask_width and image_height == mask_height:
            # mask = mask.unsqueeze(-1)  # 形状变为 [b, w, h, c]
            # 提取前三个通道,沿通道维度（dim=3）拼接
            image = torch.cat([image[..., :3], mask.unsqueeze(-1)[..., :1]], dim=3)

        image_pil = Image.fromarray(
            torch.clamp(torch.round(255.0 * image[0]), 0, 255)
            .type(torch.uint8)
            .cpu()
            .numpy()
        ).convert("RGB")

        image_in = np.array(image_pil)

        only_center_face = False
        face_helper.read_image(image_in)
        # get face landmarks for each face
        face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
        # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
        # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
        # align and warp each face
        face_helper.align_warp_face()

        # face restoration
        for cropped_face in face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                output = gfpgan_model(cropped_face_t, return_rgb=False, weight=weight)[0]
                # convert to image
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f'\tFailed inference for GFPGAN: {error}.')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face)

        # (b, h, w, c)
        face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        bgr_image = face_helper.paste_faces_to_input_image()

        image_rgb = torch.from_numpy(bgr_image.transpose(2, 0, 1)) / 255
        image_rgb = image_rgb.unsqueeze(0)
        image_rgb = image_rgb.permute(0, 2, 3, 1)  # (b, h, w, c) --> (b, c, h, w)

        torch.cuda.empty_cache()

        return (image_rgb,)


NODE_CLASS_MAPPINGS = {
    "RealESRGANModelLoader": RealESRGANModelLoader,
    "GFPGANModelLoader": GFPGANModelLoader,
    "RealESRGANImageGenerator": RealESRGANImageGenerator,
    "GFPGANImageGenerator": GFPGANImageGenerator,
    "InpaintingLamaModelLoader": InpaintingLamaModelLoader,
    "InpaintingLamaImageGenerator": InpaintingLamaImageGenerator,
    "ImageMergeGenerator": ImageMergeGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RealESRGANModelLoader": "Real-ESRGAN Model Loader",
    "GFPGANModelLoader": "GFPGAN Model Loader",
    "RealESRGANImageGenerator": "Real-ESRGAN Image Generator",
    "GFPGANImageGenerator": "GFPGAN Image Generator",
    "InpaintingLamaModelLoader": "Inpainting Lama Model Loader",
    "InpaintingLamaImageGenerator": "Inpainting Lama Image Generator",
    "ImageMergeGenerator": "Image Merge Generator",
}
