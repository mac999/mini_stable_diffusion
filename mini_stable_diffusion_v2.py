# title: mini stable diffusion
# purpose: stable diffusion architecture analysis
# date: 2024.2
# modifier: Taewook Kang
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

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=1): # 임베딩 차원, 은닉 차원, 컨텍스트 차원(self attention이면 None), 어텐션 헤드 수
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim

        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)  # Query에 대한 학습을 위한 선형 레이어
        
        if context_dim is None:
            self.self_attn = True
            self.key = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.self_attn = False
            self.key = nn.Linear(context_dim, embed_dim, bias=False)
            self.value = nn.Linear(context_dim, hidden_dim, bias=False)

    def forward(self, tokens, context=None): # 토큰들[배치, 시퀀스 크기, 은닉 차원], 컨텍스트 정보[배치, 컨텍스트 시퀀스 크기, 컨텍스트 차원]. self_attn이 True면 컨텍스트는 무시됨
        if self.self_attn: # Self-attention case
            Q = self.query(tokens)
            K = self.key(tokens)
            V = self.value(tokens)
        else: # Cross-attention case
            Q = self.query(tokens)
            K = self.key(context)
            V = self.value(context)

        # Compute score matrices, attention matrices, and context vectors
        scoremats = torch.einsum("BTH,BSH->BTS", Q, K)  # Q, K간 내적 계산. 스코어 행렬 획득
        attnmats = F.softmax(scoremats / math.sqrt(self.embed_dim), dim=-1) # 스코어 행렬의 softmax 계산
        ctx_vecs = torch.einsum("BTS,BSH->BTH", attnmats, V)  # 어텐션 행렬 적용된 V벡터 계산 
        return ctx_vecs

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim):  # 은닉 차원, 컨텍스트 차원
        super(TransformerBlock, self).__init__()

        self.attn_self = CrossAttention(hidden_dim, hidden_dim)
        self.attn_cross = CrossAttention(hidden_dim, hidden_dim, context_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 3 * hidden_dim),
            nn.GELU(),
            nn.Linear(3 * hidden_dim, hidden_dim)
        ) # Feed forward neural network. 2개의 레이어로 구성. 첫번째 레이어는 3 * hidden_dim개의 은닉 유닛을 가지고 nn.GELU 비선형성 함수를 사용. 두번째 레이어는 hidden_dim개의 은닉 유닛을 가짐

    def forward(self, x, context=None): # x: 입력 텐서[배치, 시퀀스 크기, 은닉 차원], context: 컨텍스트 텐서[배치, 컨텍스트 시퀀스 크기, 컨텍스트 차원]
        x = self.attn_self(self.norm1(x)) + x # self-attention 적용 후 layer normalization과 잔차 연결 적용
        x = self.attn_cross(self.norm2(x), context=context) + x # cross-attention 적용 후 layer normalization과 잔차 연결 적용
        x = self.ffn(self.norm3(x)) + x # feed forward neural network 적용 후 layer normalization과 잔차 연결 적용

        return x

class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super(SpatialTransformer, self).__init__()
        
        self.transformer = TransformerBlock(hidden_dim, context_dim)

    def forward(self, x, context=None): # x: 입력 텐서[배치, 채널, 높이, 너비], context: 컨텍스트 텐서[배치, 컨텍스트 시퀀스 크기, 컨텍스트 차원]
        b, c, h, w = x.shape
        x_in = x

        x = rearrange(x, "b c h w -> b (h w) c") # 입력 텐서의 차원을 재배열
        x = self.transformer(x, context) # 트랜스포머 블록 적용
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w) # 텐서의 차원을 원래대로 복원

        return x + x_in # 공간 변환기의 출력과 입력의 잔차 연결

class UNet_Tranformer(nn.Module): # 시간 의존된 스코어 기반 U-NET 모델
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256,
                 text_dim=256, nClass=10): # marginal_prob_std: 시간 t에 대한 표준편차 반환 함수, channels: 각 해상도의 특징 맵의 채널 수, embed_dim: 가우시안 랜덤 특징 임베딩의 차원, text_dim: 텍스트/숫자의 임베딩 차원, nClass: 모델링할 클래스 수
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

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.attn3 = SpatialTransformer(channels[2], text_dim)  # 컨텍스트 정보, 텍스트 임베딩 차원을 공간 트랜스포머에 설정

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.attn4 = SpatialTransformer(channels[3], text_dim)

        # 디코딩 레이어. 해상도 증가
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

        # The swish activation function
        self.act = nn.SiLU()
        self.marginal_prob_std = marginal_prob_std
        self.cond_embed = nn.Embedding(nClass, text_dim)

    def forward(self, x, t, y=None): # U-Net 아키텍처를 통과한 출력 텐서 반환. x: 입력 텐서, t: 시간 텐서, y: 타겟 텐서 (텍스트 토큰. 예. MNIST 번호). h: U-Net 아키텍처를 통과한 출력 텐서
        embed = self.act(self.time_embed(t))
        y_embed = self.cond_embed(y).unsqueeze(1)

        # Encoding path
        h1 = self.conv1(x) + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h3 = self.attn3(h3, y_embed)
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))
        h4 = self.attn4(h4, y_embed)

        # Decoding path
        h = self.tconv4(h4) + self.dense5(embed)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(h + h2) + self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)

        # Normalize output
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

def loss_fn_cond(model, x, y, marginal_prob_std, eps=1e-5): # model: 시간 의존된 스코어 기반 모델, x: 입력 데이터 미니배치, y: 조건 정보(타겟 텐서. 예. 입력 텍스트, 숫자), marginal_prob_std: 표준편차 반환 함수, eps: 수치 안정성을 위한 허용값
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps # 미니배치 크기만큼 랜덤 시간 샘플링
    z = torch.randn_like(x) # 미니배치 크기만큼 정규 분포 랜덤 노이즈 생성
    std = marginal_prob_std(random_t) # 랜덤 시간에 대한 표준편차 계산
    perturbed_x = x + z * std[:, None, None, None] # 노이즈로 입력 데이터 왜곡
    score = model(perturbed_x, random_t, y=y) # 모델을 사용해 왜곡된 데이터와 시간에 대한 스코어 획득
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3))) # score 함수와 잡음에 기반한 손실 계산
    return loss

# 데이터 학습
score_model = torch.nn.DataParallel(UNet_Tranformer(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

n_epochs = 100   
batch_size = 1024 
lr = 10e-4        

transform = transforms.Compose([
    transforms.ToTensor()  # Convert image to PyTorch tensor
])
dataset = torchvision.datasets.MNIST('.', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = Adam(score_model.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0

    for x, y in tqdm(data_loader):
        x = x.to(device)

        loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]

    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

    torch.save(score_model.state_dict(), 'ckpt_transformer.pth')

# MNIST 생성AI 테스트
ckpt = torch.load('ckpt_transformer.pth', map_location=device)
score_model.load_state_dict(ckpt)

digit = 9 # 생성AI 입력 조건. 0~9까지 숫자 생성 가능

sample_batch_size = 64 
num_steps = 250
sampler = Euler_Maruyama_sampler
# score_model.eval()

samples = sampler(score_model,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        sample_batch_size,
        num_steps=num_steps,
        device=device,
        y=digit*torch.ones(sample_batch_size, dtype=torch.long))

# 생성 결과 확인
samples = samples.clamp(0.0, 1.0)
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()
