import cv2
import os
import numpy as np
import torch
from .basicsr.rrdbnet_arch import RRDBNet


def load_model(model_path):
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # prefer to use params_ema
    if 'params_ema' in state_dict:
        keyname = 'params_ema'
    else:
        keyname = 'params'

    net.load_state_dict(state_dict[keyname], strict=True)
    net.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = net.to(device)
    model = model.half()

    return model


@torch.no_grad()
def image_gan(model, image_path, image_output):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_width, image_height = image.shape[0:2]
    name = os.path.basename(image_path)

    print(name, image.shape, image_height, image_width)

    scale = 4
    alpha_upsampler = 'realesrgan'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = image.astype(np.float32)
    if np.max(image) > 256:  # 16-bit image
        max_range = 65535
        print('\tInput is a 16-bit image')
    else:
        max_range = 255

    image = image / max_range
    if len(image.shape) == 2:  # gray image
        img_mode = 'L'
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA image with alpha channel
        img_mode = 'RGBA'
        alpha = image[:, :, 3]
        image = image[:, :, 0:3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if alpha_upsampler == 'realesrgan':
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    else:
        img_mode = 'RGB'
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("image.shape f ", image.shape)
    image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
    image = image.unsqueeze(0).to(device)
    image = image.half()

    print("image.shape b ", image.shape)

    result = model(image)
    print("result", result.shape)

    output_img = result.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
    if img_mode == 'L':
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    print("img_mode", img_mode)
    # ------------------- process the alpha channel if necessary ------------------- #
    if img_mode == 'RGBA':
        if alpha_upsampler == 'realesrgan':

            print("alpha.shape f", alpha.shape)
            alpha = torch.from_numpy(np.transpose(alpha, (2, 0, 1))).float()
            alpha = alpha.unsqueeze(0).to(device)
            alpha = alpha.half()
            print("alpha.shape b", alpha.shape)
            output_alpha = model(alpha)

            print("result alpha ", output_alpha.shape)
            output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
            output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
        else:  # use the cv2 resize for alpha channel
            h, w = alpha.shape[0:2]
            output_alpha = cv2.resize(alpha, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

        # merge the alpha channel
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha

    # ------------------------------ return ------------------------------ #
    if max_range == 65535:  # 16-bit image
        output = (output_img * 65535.0).round().astype(np.uint16)
    else:
        output = (output_img * 255.0).round().astype(np.uint8)

    if os.path.isdir(image_output):
        image_output = os.path.join(image_output, "{}_out.{}".format(name.split(".")[0], name.split(".")[1]))

    print("imwrite", image_output, output.shape)
    cv2.imwrite(image_output, output)


"""
enhance (755, 1425, 4)
enhance (755, 1425, 3)
pre_process torch.Size([1, 3, 755, 1425]) 0

(179, 179, 3) 179 179
(179, 179, 3)
torch.Size([1, 3, 179, 179])
torch.Size([1, 3, 716, 716])


00003.png
(256, 512, 3) 512 256
(256, 512, 3)
torch.Size([1, 3, 256, 512])
torch.Size([1, 3, 1024, 2048])

00003
process  torch.Size([1, 3, 256, 512]) torch.Size([1, 3, 1024, 2048])
00017_gray
process  torch.Size([1, 3, 351, 500]) torch.Size([1, 3, 1404, 2000])
0014
process  torch.Size([1, 3, 179, 179]) torch.Size([1, 3, 716, 716])
0030
process  torch.Size([1, 3, 220, 220]) torch.Size([1, 3, 880, 880])

dd4ed4da-e461-4ca1-b0bd-b48b77513d0e
process  torch.Size([1, 3, 755, 1425]) torch.Size([1, 3, 3020, 5700])


"""


def test():
    image_path = "C:\\Users\\Administrator\\Pictures/353211d6-b862-4e9b-9171-419363e3b167.jpeg"
    model_name = "RealESRGAN_x4plus"

    image_output = "D:\\ComfyUI\\temp"
    model_path = os.path.join(os.path.join("D:\\ComfyUI\\models", "Hunyuan3D-2"), "{}.pth".format(model_name))
    # model_path = "./weights/RealESRGAN_x4plus.pth"
    model = load_model(model_path)

    image_gan(model, image_path, image_output)
    # print(model)


if __name__ == '__main__':
    test()
