import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy  # For KL divergence calculation

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_weights

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None,label_prefix=''):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'img')
        self.label_dir = os.path.join(root_dir, 'txt')
        self.transform = transform
        self.label_prefix = label_prefix  # 每个数据集的唯一前缀
        self.image_paths = sorted([os.path.join(self.image_dir, fname) for fname in os.listdir(self.image_dir)])
        self.label_paths = sorted([os.path.join(self.label_dir, fname) for fname in os.listdir(self.label_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        with open(label_path, 'r') as f:
            label = f.readline().strip().split(' ')[0]  # 读取标签中的id部分
        label = f"{self.label_prefix}{label}"  # 添加前缀
        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# 加载不同数据集
dataset1 = CustomDataset(root_dir='/home/lenovo/data/liujiaji/Datasets/Einsulator/defect/', transform=transform,label_prefix='E1-')
dataset2 = CustomDataset(root_dir='/home/lenovo/data/liujiaji/Datasets/CPLID/Defective_Insulators/', transform=transform,label_prefix='C1-')
dataset3 = CustomDataset(root_dir='/home/lenovo/data/liujiaji/Datasets/VPMBGI/', transform=transform,label_prefix='B1-')
dataset4 = CustomDataset(root_dir='/home/lenovo/data/liujiaji/Datasets/Einsulator/burn/', transform=transform,label_prefix='E2-')
dataset5 = CustomDataset(root_dir='/home/lenovo/data/liujiaji/Datasets/IDID/train/', transform=transform,label_prefix='I1-')

# 合并数据集
combined_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3, dataset4, dataset5])
dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=True, num_workers=4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 其他hub可下载的预训练模型，v5,6,7
# model = torch.hub.load('ultralytics/yolov5', 'yolov5x' ,pretrained=True).to(device)
# model.eval()

# v8 参照heatmap
# weight = '/home/lenovo/data/liujiaji/yolov8/ultralytics-main/yolov8m.pt'
weight = 'runs/train/exp112/weights/best.pt'
model = attempt_load_weights(weight, device)
# print(model)
model.eval()


# 特征提取
features = []
labels = []
for images, lbls in tqdm(dataloader, desc='Running the model inference'):
    images = images.to(device)

    # 其他hub可下载的预训练模型，v5,6,7
    # output = model(images) # torch.Size([8, 25200, 85])
 
    output = model(images)[0] # torch.Size([8, 84, 8400])
    features.append(output.cpu().detach().numpy())
    labels.extend(lbls)

print('1',len(features))  # 180

features = np.concatenate(features, axis=0)
print('2',len(features))  # 1440
print('2',features.shape)  # (1440, 84, 8400)

features = np.array(features).reshape(len(features), -1)
print('3',len(features))  # 1440
print('3',features.shape)  # (1440, 705600)


# 打印特征和标签的长度
print(f'Number of features: {len(features)}') # 1440
print(f'Number of labels: {len(labels)}') # 1440

# 确保特征和标签数量一致
assert len(features) == len(labels), "Features and labels length mismatch!"


# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features) 
print(tsne_results.shape) #(1440,2)

# 映射标签到新名称
label_mapping = {
    'E1-1': 'insulator-defect',
    'B1-4': 'VPMBGI-defect',
    'C1-defect': 'CPLID-defect',
    'I1-3': 'IDID-broken',

    'E2-0': 'insulator-burn',
    'I1-4': 'IDID-flashover',

}

# 将标签映射到新标签名
mapped_labels = [label_mapping.get(str(label), str(label)) for label in labels]

# 选择要可视化的标签
# visualize_labels = ['insulator-defect', 'VPMBGI-defect','CPLID-defect','IDID-broken']  # 仅可视化标签为 
visualize_labels = ['insulator-burn', 'IDID-flashover']

# 可视化
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

tx = tsne_results[:, 0]
ty = tsne_results[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# 不同数据集标签的不同样式
styles = {
    'insulator-defect': ('o', 'lightsalmon'),
    'CPLID-defect':('+','cyan'),
    'VPMBGI-defect':('*','seagreen'),
    'IDID-broken': ('^', 'skyblue'),

    'insulator-burn': ('s', 'green'),
    'IDID-flashover': ('D', 'pink'),
    

}

# 绘制2D点，每个点的颜色与类标签对应
fig, ax = plt.subplots()
unique_labels = np.unique(mapped_labels)
# print(unique_labels)
colors_per_class = {label: plt.cm.tab10(i % 10) for i, label in enumerate(unique_labels)}

# 计算每个类的均值坐标
class_means = {}

for label in unique_labels:
    if label in visualize_labels: 
        indices = [i for i, l in enumerate(mapped_labels) if l == label] # i是标签索引，l是标签值
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = colors_per_class[label]
        # new_label = label_mapping.get(label, label)  # 映射到新标签名
        # # print(f"Original label: {label}, Mapped label: {new_label}")
        # ax.scatter(current_tx, current_ty, c=[color], label=label, alpha=0.9, edgecolors='k', linewidth=0.9)

        # marker, color = styles.get(label, ('o', 'black'))  # 默认样式为黑色圆点
        # ax.scatter(current_tx, current_ty, c=color, marker=marker, label=label, alpha=0.9, edgecolors='black', linewidth=0.8)

        #  计算每个类的均值坐标，之后 计算不同数据分布间的距离
        if indices:  # 确保该类有样本
            class_means[label] = np.mean(tsne_results[indices], axis=0)

# 可视化TSNE结果
# ax.legend(loc='best')
# plt.title('t-SNE visualization of the dataset')
# plt.savefig('/home/lenovo/data/liujiaji/powerGit/dataset/tsne-defect-ours-1.jpg')

# 将均值转换为 numpy 数组
mean_array = np.array(list(class_means.values()))
mean_labels = list(class_means.keys())

# 计算均值之间的欧氏距离
euclidean_distances_matrix = pairwise_distances(mean_array)
# print("Euclidean Distances between class means:\n", euclidean_distances_matrix)

# 计算均值之间的余弦相似度
cosine_similarities_matrix = cosine_similarity(mean_array)
# print("Cosine Similarities between class means:\n", cosine_similarities_matrix)

# 计算均值之间的 KL 散度
kl_divergence_matrix = np.zeros((len(mean_labels), len(mean_labels)))
for i in range(len(mean_labels)):
    for j in range(len(mean_labels)):
        if i != j:
            # 通过简单的归一化处理来创建概率分布
            p = np.exp(mean_array[i] - np.max(mean_array[i]))  # 防止溢出
            q = np.exp(mean_array[j] - np.max(mean_array[j]))  # 防止溢出
            p /= np.sum(p)
            q /= np.sum(q)
            kl_divergence_matrix[i, j] = entropy(p, q)  # KL散度
# print("KL Divergence between class means:\n", kl_divergence_matrix)


from scipy.spatial import distance
from scipy.special import kl_div
import seaborn as sns

# 计算马氏距离
# mahalanobis_distances = distance.cdist(mean_array, mean_array, metric='mahalanobis')
# print("\nMahalanobis Distances between class means:\n", mahalanobis_distances)

# 计算 Jensen-Shannon Divergence
js_divergence_matrix = np.zeros((len(mean_labels), len(mean_labels)))
for i in range(len(mean_labels)):
    for j in range(len(mean_labels)):
        if i != j:
            # 通过简单的归一化处理来创建概率分布
            p = np.exp(mean_array[i] - np.max(mean_array[i]))  # 防止溢出
            q = np.exp(mean_array[j] - np.max(mean_array[j]))  # 防止溢出
            p /= np.sum(p)
            q /= np.sum(q)
            m = 0.5 * (p + q)  # 混合分布
            js_divergence_matrix[i, j] = 0.5 * (entropy(p, m) + entropy(q, m))  # Jensen-Shannon divergence
print("Jensen-Shannon Divergence between class means:\n", js_divergence_matrix)

# 绘制距离矩阵热图
plt.figure(figsize=(12, 8))
sns.heatmap(euclidean_distances_matrix, annot=True, fmt=".4f", cmap='coolwarm', xticklabels=mean_labels, yticklabels=mean_labels)
plt.title('Euclidean Distances Heatmap')
plt.savefig('/home/lenovo/data/liujiaji/powerGit/dataset/burn-euclidean_heatmap.jpg')

plt.figure(figsize=(12, 8))
sns.heatmap(cosine_similarities_matrix, annot=True, fmt=".4f", cmap='coolwarm', xticklabels=mean_labels, yticklabels=mean_labels)
plt.title('Cosine Similarities Heatmap')
plt.savefig('/home/lenovo/data/liujiaji/powerGit/dataset/burn-cosine_heatmap.jpg')

plt.figure(figsize=(12, 8))
sns.heatmap(kl_divergence_matrix, annot=True, fmt=".4f", cmap='coolwarm', xticklabels=mean_labels, yticklabels=mean_labels)
plt.title('KL Divergence Heatmap')
plt.savefig('/home/lenovo/data/liujiaji/powerGit/dataset/burn-kl_heatmap.jpg')

plt.figure(figsize=(12, 8))
sns.heatmap(js_divergence_matrix, annot=True, fmt=".4f", cmap='coolwarm', xticklabels=mean_labels, yticklabels=mean_labels)
plt.title('Jensen-Shannon Divergence Heatmap')
plt.savefig('/home/lenovo/data/liujiaji/powerGit/dataset/burn-js_heatmap.jpg')

# plt.figure(figsize=(12, 8))
# sns.heatmap(mahalanobis_distances, annot=True, fmt=".4f", cmap='coolwarm', xticklabels=mean_labels, yticklabels=mean_labels)
# plt.title('Mahalanobis Distances Heatmap')
# plt.savefig('/home/lenovo/data/liujiaji/powerGit/dataset/burn-mahalanobis_heatmap.jpg')



# for i, label_i in enumerate(mean_labels):
#     for j, label_j in enumerate(mean_labels):
#         if i != j:
#             print(f"Distance between {label_i} and {label_j}: {euclidean_distances_matrix[i, j]:.4f}")
#             print(f"Cosine similarity between {label_i} and {label_j}: {cosine_similarities_matrix[i, j]:.4f}")
#             print(f"KL Divergence between {label_i} and {label_j}: {kl_divergence_matrix[i, j]:.4f}")

# 可视化均值的散点图
mean_tx, mean_ty = mean_array[:, 0], mean_array[:, 1]
# 绘制均值点
for i, label in enumerate(mean_labels):
    plt.scatter(mean_tx[i], mean_ty[i], color=colors_per_class[label], marker='X', s=200, label=f'Mean of {label}')
plt.title('Class Means in t-SNE Space')
plt.legend()
plt.savefig('/home/lenovo/data/liujiaji/powerGit/dataset/tsne-burn-distance.jpg')

'''
burn:
Euclidean Distances between class means:
            IDID        insu
IDID[[          0      16.508]
insu [     16.508           0]]
Cosine Similarities between class means:
 [[          1     0.88657]
 [    0.88657           1]]
KL Divergence between class means:
 [[          0  0.00015351]
 [ 0.00017933           0]]


defect:
Euclidean Distances between class means: 值最小，距离越近
                CPLID       IDID       VPMBGI        insu
 CPLID  [[          0      52.251      33.575      33.089]
 IDID   [     52.251           0      20.661      21.674]
 VPMBGI [     33.575      20.661           0      1.1976]
 insu   [     33.089      21.674      1.1976           0]]
Cosine Similarities between class means:  值最大，越相似，值为0时，既不相似也不同
 [[          1    -0.94839     0.87883     0.85436]
 [   -0.94839           1    -0.98478    -0.97506]
 [    0.87883    -0.98478           1      0.9988]
 [    0.85436    -0.97506      0.9988           1]]
KL Divergence between class means: 值越小，分布越相似
 [[          0      6.2946  0.00028651   0.0010506]
 [     7.3931           0      8.5577      10.073]
 [ 0.00019501      6.3005           0  0.00013765]
 [ 0.00044862      6.3027  8.3573e-05           0]]


'''



