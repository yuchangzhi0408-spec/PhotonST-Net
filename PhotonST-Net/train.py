import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import UNetWithDynamicOutputs
from pytorch_msssim import ssim

# 清空显存缓存
torch.cuda.empty_cache()

# 路径设置
input_root_directory = './data/input'  # 输入图像
target_directory = './data/target'       # 标准图像
model_directory = './checkpoints'       # 模型保存路径
csv_path = os.path.join(model_directory, 'loss_log.csv')

os.makedirs(model_directory, exist_ok=True)

# 参数设置
batch_size = 16
frame_range = 1  # 帧范围：前后各1帧，共3帧输入
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

epochs = 100
learning_rate = 0.001
alpha, beta = 0.7, 0.3  # α·MSE + β·(1-SSIM)

# 损失函数定义
def ssim_loss(img1, img2):
    return 1 - ssim(img1, img2, data_range=1.0, size_average=True)

# 添加自定义的collate函数
def custom_collate(batch):
    # 分离输入和目标
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # 处理输入 - 它们都是相同大小的张量
    inputs = torch.stack(inputs)
    
    # 确定这个批次是单目标还是双目标
    max_targets = max(len(t) for t in targets)
    
    # 处理目标
    processed_targets = []
    for i in range(max_targets):
        target_batch = []
        for t in targets:
            if i < len(t):
                target_batch.append(t[i])
            else:
                # 如果这个样本没有第二个目标，用零张量填充
                target_batch.append(torch.zeros_like(t[0]))
        processed_targets.append(torch.stack(target_batch))
    
    return inputs, processed_targets

# 修改损失函数以处理填充的目标
def combined_loss(output1, output2, class_pred, target_list, alpha, beta):
    loss = 0

    # 计算分类损失
    if len(target_list) == 1:  # 单目标情况
        target_class = torch.zeros(output1.size(0), dtype=torch.long, device=output1.device)
    else:  # 双目标情况
        is_dual = ~(target_list[1] == 0).all(1).all(1)
        target_class = is_dual.long()

    class_loss = F.cross_entropy(class_pred, target_class)

    # 计算重建损失: α·MSE + β·(1-SSIM)
    batch_size = output1.size(0)
    recon_loss = 0

    for i in range(batch_size):
        if len(target_list) == 1:  # 单目标情况
            target = target_list[0][i:i+1].unsqueeze(1)
            ssim_val = ssim_loss(output1[i:i+1], target)
            mse_val = F.mse_loss(output1[i:i+1], target)
            recon_loss += alpha * mse_val + beta * ssim_val
        else:  # 双目标情况
            if not (target_list[1][i] == 0).all():
                target1 = target_list[0][i:i+1].unsqueeze(1)
                ssim_val1 = ssim_loss(output1[i:i+1], target1)
                mse_val1 = F.mse_loss(output1[i:i+1], target1)

                target2 = target_list[1][i:i+1].unsqueeze(1)
                ssim_val2 = ssim_loss(output2[i:i+1], target2)
                mse_val2 = F.mse_loss(output2[i:i+1], target2)

                recon_loss += (alpha * (mse_val1 + mse_val2) +
                              beta * (ssim_val1 + ssim_val2)) / 2
            else:
                target = target_list[0][i:i+1].unsqueeze(1)
                ssim_val = ssim_loss(output1[i:i+1], target)
                mse_val = F.mse_loss(output1[i:i+1], target)
                recon_loss += alpha * mse_val + beta * ssim_val

    recon_loss = recon_loss / batch_size

    # 总损失: Loss_recon + λ·Loss_class, λ=0.15
    recon_weight = 1.0
    class_weight = 0.15
    total_loss = recon_weight * recon_loss + class_weight * class_loss
    return total_loss

# 创建或更新 CSV 文件
def update_csv(csv_path, epoch, train_loss, test_loss):
    new_row = pd.DataFrame({'Epoch': [epoch], 'Train Loss': [train_loss], 'Test Loss': [test_loss]})
    if not os.path.exists(csv_path):
        new_row.to_csv(csv_path, index=False)
    else:
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
        combined.to_csv(csv_path, index=False)

# 数据集定义
class FrameDataset(Dataset):
    def __init__(self, input_dir, target_dir, frame_range=1, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.frame_range = frame_range
        self.transform = transform
        self.folders = sorted([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])
        self._cache = {}
        
        # 检查每个文件夹的目标类型
        self.folder_types = {}  # 存储每个文件夹是单目标还是双目标
        for folder in self.folders:
            target_folder = os.path.join(target_dir, folder)
            if os.path.exists(os.path.join(target_folder, '1')) and os.path.exists(os.path.join(target_folder, '2')):
                self.folder_types[folder] = 2  # 双目标
            else:
                self.folder_types[folder] = 1  # 单目标

    def __len__(self):
        return sum(
            max(0, len(os.listdir(os.path.join(self.input_dir, folder))) - 2 * self.frame_range)
            for folder in self.folders
        )

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        folder_idx = 0
        while idx >= len(os.listdir(os.path.join(self.input_dir, self.folders[folder_idx]))) - 2 * self.frame_range:
            idx -= len(os.listdir(os.path.join(self.input_dir, self.folders[folder_idx]))) - 2 * self.frame_range
            folder_idx += 1

        folder_name = self.folders[folder_idx]
        img_list = sorted([f for f in os.listdir(os.path.join(self.input_dir, folder_name)) if
                           f.endswith('.png') or f.endswith('.bmp')])

        local_idx = idx + self.frame_range
        frame_list = []

        for i in range(-self.frame_range, self.frame_range + 1):
            img_name = img_list[max(0, min(local_idx + i, len(img_list) - 1))]  # 保证不会越界
            img = Image.open(os.path.join(self.input_dir, folder_name, img_name)).convert('L')
            if self.transform:
                img = self.transform(img)
            frame_list.append(img)

        input_tensor = torch.cat(frame_list, dim=0)

        # 根据文件夹类型获取目标
        target_images = []
        target_folder = os.path.join(self.target_dir, folder_name)
        
        if self.folder_types[folder_name] == 1:  # 单目标
            target_img = Image.open(os.path.join(target_folder, img_list[local_idx])).convert('L')
            if self.transform:
                target_img = self.transform(target_img)
            target_images.append(target_img.squeeze(0))  # 移除通道维度，在损失函数中再添加
        else:  # 双目标
            for class_idx in range(2):
                subfolder = os.path.join(target_folder, str(class_idx + 1))
                target_img = Image.open(os.path.join(subfolder, img_list[local_idx])).convert('L')
                if self.transform:
                    target_img = self.transform(target_img)
                target_images.append(target_img.squeeze(0))  # 移除通道维度，在损失函数中再添加

        result = (input_tensor, target_images)
        self._cache[idx] = result
        return result

# 数据预处理
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# 数据加载
dataset = FrameDataset(input_root_directory, target_directory, frame_range, transform)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 修改数据加载器的创建
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate  # 使用自定义的collate函数
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate  # 使用自定义的collate函数
)

# 模型和优化器
model = UNetWithDynamicOutputs(n_channels=1 + 2 * frame_range).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 添加学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=True
)

# 初始化最佳损失和最佳epoch
best_loss = float('inf')
best_epoch = -1  # 记录最佳表现的epoch

# 训练过程
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Training", leave=True, ncols=100)
    for input_tensors, target_list in pbar:
        input_tensors = input_tensors.to(device, non_blocking=True)
        target_list = [target.to(device, non_blocking=True) for target in target_list]

        outputs1, outputs2, _, _, class_pred = model(input_tensors)
        loss = combined_loss(outputs1, outputs2, class_pred, target_list, alpha, beta)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    train_loss_avg = train_loss / len(train_loader)

    # 测试阶段
    model.eval()
    test_loss = 0.0
    pbar = tqdm(test_loader, desc=f"Epoch {epoch}/{epochs} Testing", leave=True, ncols=100)
    with torch.no_grad():
        for input_tensors, target_list in pbar:
            input_tensors = input_tensors.to(device, non_blocking=True)
            target_list = [target.to(device, non_blocking=True) for target in target_list]

            outputs1, outputs2, _, _, class_pred = model(input_tensors)
            loss = combined_loss(outputs1, outputs2, class_pred, target_list, alpha, beta)
            test_loss += loss.item()

    test_loss_avg = test_loss / len(test_loader)
    scheduler.step(test_loss_avg)  # 根据验证损失调整学习率

    # 在每个epoch结束后打印训练和测试的平均损失
    print(f"\nEpoch [{epoch}/{epochs}]")
    print(f"Training Loss: {train_loss_avg:.4f}")
    print(f"Testing Loss: {test_loss_avg:.4f}")

    # 更新CSV文件
    update_csv(csv_path, epoch, train_loss_avg, test_loss_avg)

    # 保存每个epoch的模型
    epoch_model_path = os.path.join(model_directory, f'epoch_{epoch}.pth')
    torch.save(model.state_dict(), epoch_model_path)

    # 保存最佳模型
    if test_loss_avg < best_loss:
        best_loss = test_loss_avg
        best_epoch = epoch
        best_model_path = os.path.join(model_directory, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"Model improved! Best loss: {best_loss:.4f}, saved as {best_model_path}")
    
    print(f"Best epoch so far: {best_epoch} with Test Loss = {best_loss:.4f}\n")

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
