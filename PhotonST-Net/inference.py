import os
import torch
from torchvision import transforms
from PIL import Image
from model import UNetWithDynamicOutputs

torch.cuda.empty_cache()

# 路径设置
input_dir = './data/test'       # 测试图像
output_base_dir = './output'    # 输出目录
model_path = './checkpoints/best_model.pth'  # 模型权重

# 参数设置
frame_range = 1  # 前后各1帧，共3帧输入
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNetWithDynamicOutputs(n_channels=1 + 2 * frame_range).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

os.makedirs(os.path.join(output_base_dir, '1'), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, '2'), exist_ok=True)

img_list = sorted(os.listdir(input_dir))

for i in range(frame_range, len(img_list) - frame_range):
    frame_list = []
    for j in range(-frame_range, frame_range + 1):
        img = Image.open(os.path.join(input_dir, img_list[i + j])).convert('L')
        img = transform(img)
        frame_list.append(img)

    input_tensor = torch.cat(frame_list, dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        output1, output2, _, _, class_pred = model(input_tensor)

        is_dual_frame = class_pred[0, 1] > 0.5

        output1 = output1.squeeze(0).squeeze(0).cpu()
        output_img1 = transforms.ToPILImage()(output1)
        output_img1.save(os.path.join(output_base_dir, '1', img_list[i]))

        if is_dual_frame:
            output2 = output2.squeeze(0).squeeze(0).cpu()
            output_img2 = transforms.ToPILImage()(output2)
            output_img2.save(os.path.join(output_base_dir, '2', img_list[i]))

    del output1, output2
    torch.cuda.empty_cache()

print("所有图片已处理完毕！")
