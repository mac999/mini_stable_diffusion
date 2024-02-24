# 스테이블 디퓨전 모델 정의
import math, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
import functools, torchvision, torchvision.transforms as transforms
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR
from torchvision.utils import make_grid
from einops import rearrange

device = "cuda"

# 가우시안 푸리에 특징 임베딩 계층 정의
class GaussianFourierProjection(nn.Module):  # 시간 특징 계산 클래스
    def __init__(self, embed_dim, scale=30.): # 임베딩 차원, 랜덤 가중치(주파수)를 위한 스케일 변수
        super().__init__()

        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)  # 랜덤 샘플링. 훈련 파라메터 아님.

    def forward(self, x): # 매 시간단위 텐서 입력
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi  # Cosine(2 pi freq x), Sine(2 pi freq x)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # 최종 sine, cosine 결과 벡터 결합

class Dense(nn.Module): # 특징 계산 클래스
    def __init__(self, input_dim, output_dim):  # 입력 특징 차원, 출력 특징 차원
        super().__init__()

        self.dense = nn.Linear(input_dim, output_dim) # 선형 전연결 레이어

    def forward(self, x): # 입력 텐서
        return self.dense(x)[..., None, None]  # 학습 후 4D텐서

# U-Net 모델 정의
class UNet(nn.Module): # U-Net 모델 정의. nn.Module 클래스 상속
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):  # marginal_prob_std: 시간 t에 대한 표준편차 반환 함수, channels: 각 해상도의 특징 맵의 채널 수, embed_dim: 가우시안 랜덤 특징 임베딩의 차원

        super().__init__()

        # 시간에 대한 가우시안 랜덤 특징 임베딩 계층
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # 인코딩 레이어 (해상도 감소)
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        # 추가 인코딩 레이어 (원본 코드에서 복사)
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # 해상도 증가를 위한 디코딩 레이어 
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1)

        # 스위치 활성화 함수
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, y=None):    # U-Net 아키텍처를 통과한 출력 텐서 반환. x: 입력 텐서, t: 시간 텐서, y: 타겟 텐서 (이 전방 통과에서 사용되지 않음). h: U-Net 아키텍처를 통과한 출력 텐서
        # t에 대한 가우시안 랜덤 특징 임베딩 획득
        embed = self.act(self.time_embed(t))

        # 인코딩 경로
        h1 = self.conv1(x) + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))

        # 추가 인코딩 경로 레이어 (원본 코드에서 복사)
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))

        # 디코딩 경로
        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3)
        h += self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(h + h2)
        h += self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)

        # 정규화 출력
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           x_shape=(1, 28, 28),
                           num_steps=500,
                           device='cuda',
                           eps=1e-3, y=None): # Euler-Maruyama sampler 함수. score-based 모델을 사용해 샘플 생성. score_model: 시간 의존 스코어 모델, marginal_prob_std: 표준편차 반환 함수, diffusion_coeff: 확산 계수 함수, batch_size: 한번 호출시 생성할 샘플러 수, x_shape: 샘플 형태, num_steps: 샘플링 단계 수, device: 'cuda' 또는 'cpu', eps: 수치 안정성을 위한 허용값, y: 타겟 텐서 (이 함수에서 사용되지 않음) 

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, *x_shape, device=device) * marginal_prob_std(t)[:, None, None, None]  # Initial sample
    
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1] # Step size 시리즈
    x = init_x # 시간 t에 대한 초기 샘플
    
    with torch.no_grad(): # Euler-Maruyama 샘플링
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    
    return mean_x

# 데이터 학습 준비
def marginal_prob_std(t, sigma): # 시간 t에 대한 표준편차 반환 함수
    t = torch.tensor(t, device=device) # 시간 텐서    
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma)) # 시간 t에 대한 표준편차 반환

def diffusion_coeff(t, sigma): # 확산 계수 함수. t: 시간 텐서, sigma: SDE의 시그마
    return torch.tensor(sigma**t, device=device) # 확산 계수 반환

sigma =  25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

def loss_fn(model, x, marginal_prob_std, eps=1e-5): # 손실함수. score-based generative models 훈련용. model: 시간 의존 스코어 모델, x: 훈련 데이터 미니배치, marginal_prob_std: 표준편차 반환 함수, eps: 수치 안정성을 위한 허용값
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - 2 * eps) + eps # 미니배치 크기만큼 랜덤 시간 샘플링

    std = marginal_prob_std(random_t) # 랜덤 시간에 대한 표준편차 계산
    z = torch.randn_like(x) # 미니배치 크기만큼 정규 분포 랜덤 노이즈 생성    
    perturbed_x = x + z * std[:, None, None, None] # 노이즈로 입력 데이터 왜곡

    score = model(perturbed_x, random_t) # 모델을 사용해 왜곡된 데이터와 시간에 대한 스코어 획득    
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3))) # score 함수와 잡음에 기반한 손실 계산
    
    return loss

score_model = torch.nn.DataParallel(UNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

batch_size = 1024
transform = transforms.Compose([
    transforms.ToTensor()  # Convert image to PyTorch tensor
    # transforms.Normalize((0.5,), (0.5,))  # Normalize images with mean and std dev
])
dataset = torchvision.datasets.MNIST('.', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 데이터 학습
n_epochs = 75
lr = 1e-3

optimizer = Adam(score_model.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    # for i, data in enumerate(data_loader, 0):
    for x, y in tqdm(data_loader):
        x = x.to(device)
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]

    print(f'Epoch {epoch}, Average Loss: {avg_loss / num_items:5f}')
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

    torch.save(score_model.state_dict(), 'ckpt.pth')    # save every epoch

# 샘플링
ckpt = torch.load('ckpt.pth', map_location=device)
score_model.load_state_dict(ckpt)

sample_batch_size = 64
num_steps = 500

sampler = Euler_Maruyama_sampler  # Euler-Maruyama sampler 사용
samples = sampler(score_model, marginal_prob_std_fn,
                  diffusion_coeff_fn, sample_batch_size,
                  num_steps=num_steps, device=device, y=None)
samples = samples.clamp(0.0, 1.0)

import matplotlib.pyplot as plt
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()    