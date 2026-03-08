# %% [markdown]
# # Задание 8 (базовый вариант)
#
# **Вариант:** базовый
#
# Требования:
# 1) Расписать код (полный, читаемый, с комментариями).
# 2) Ниже добавить текстовую ячейку с описанием, как работает код (запускать не обязательно).
# 3) Построить 2 диаграммы GAN: Generator и Discriminator (расположить над кодом моделей).
#
# Ниже — ноутбук-скрипт в стиле VS Code (Python extension) с разделением на блоки `# %%`.
#

# %%
from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image


# %%
# -----------------------------
# Конфигурация / воспроизводимость
# -----------------------------

@dataclass(frozen=True)
class Config:
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 2
    image_size: int = 28            # MNIST: 28x28
    channels: int = 1               # MNIST: grayscale
    latent_dim: int = 100           # размер шума z
    g_features: int = 64            # базовая ширина генератора
    d_features: int = 64            # базовая ширина дискриминатора
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    epochs: int = 20
    out_dir: str = "artifacts"
    fixed_noise_n: int = 64         # сколько изображений генерировать для мониторинга


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Для детерминизма (может замедлять)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


cfg = Config()
seed_everything(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(cfg.out_dir, exist_ok=True)

print("device:", device)


# %%
# -----------------------------
# Данные: MNIST
# -----------------------------
# Нормировка в диапазон [-1, 1] удобна для генератора с tanh на выходе.

transform = T.Compose([
    T.Resize(cfg.image_size),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),  # (x - 0.5)/0.5 -> [-1,1]
])

train_ds = torchvision.datasets.MNIST(
    root=os.path.join(cfg.out_dir, "data"),
    train=True,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=(device.type == "cuda"),
)

# Быстрая проверка батча
x0, y0 = next(iter(train_loader))
print("batch:", x0.shape, y0.shape, "min/max:", float(x0.min()), float(x0.max()))


# %% [markdown]
# ## Диаграммы GAN (Generator / Discriminator)
# Ниже — блок, который **создаёт диаграммы** вычислительного графа для генератора и дискриминатора и сохраняет их в файлы.
# Это нужно выполнить **до** блока с кодом моделей (как требуется в задании).
#
# Диаграммы строятся через `torchviz` (Graphviz). Если библиотек нет — будут напечатаны инструкции.
#

# %%
# -----------------------------
# Утилиты для диаграмм
# -----------------------------
def try_make_torchviz_graph(
    model: nn.Module,
    example_input: torch.Tensor,
    filename_no_ext: str,
    out_dir: str,
) -> None:
    """
    Пытается построить диаграмму графа через torchviz и сохранить:
    - <filename_no_ext>.png
    - <filename_no_ext>.dot (если доступно)
    """
    try:
        from torchviz import make_dot  # type: ignore
    except Exception as e:
        print(f"[torchviz] недоступен ({e}). Установка (Colab):")
        print("  !pip install torchviz graphviz")
        print("  !apt-get update && apt-get install -y graphviz")
        return

    model.eval()
    with torch.no_grad():
        out = model(example_input)

    dot = make_dot(out, params=dict(model.named_parameters()))
    # torchviz рендерит через graphviz; формат PNG
    dot.format = "png"
    out_path = os.path.join(out_dir, filename_no_ext)
    dot.render(out_path, cleanup=True)
    print(f"[torchviz] сохранено: {out_path}.png")


# %%
# -----------------------------
# Модели GAN (DCGAN-подобные под MNIST 28x28)
# -----------------------------

class Generator(nn.Module):
    """
    Генератор: z -> изображение [B, C, H, W] в диапазоне [-1, 1].
    Архитектура: MLP + ConvTranspose2d блоки.
    """
    def __init__(self, latent_dim: int, channels: int, features: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Проекция из latent в "карты признаков" 7x7
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, features * 4 * 7 * 7),
            nn.BatchNorm1d(features * 4 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        # Апсемплинг: 7x7 -> 14x14 -> 28x28
        self.net = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(features * 2, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim]
        x = self.proj(z)                    # [B, f*4*7*7]
        x = x.view(z.size(0), -1, 7, 7)     # [B, f*4, 7, 7]
        img = self.net(x)                   # [B, C, 28, 28]
        return img


class Discriminator(nn.Module):
    """
    Дискриминатор: изображение -> логит (не sigmoid).
    Используем BCEWithLogitsLoss.
    """
    def __init__(self, channels: int, features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 14x14 -> 7x7
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 7x7 -> 4x4 (stride=2, pad=1, k=4 => floor((7+2-4)/2)+1 = 4)
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear((features * 4) * 4 * 4, 1),  # логит
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        logit = self.head(h)
        return logit


# %%
# Инициализация моделей

G = Generator(cfg.latent_dim, cfg.channels, cfg.g_features).to(device)
D = Discriminator(cfg.channels, cfg.d_features).to(device)

print("G params:", sum(p.numel() for p in G.parameters()))
print("D params:", sum(p.numel() for p in D.parameters()))

# %%
# -----------------------------
# Диаграммы моделей (сохранение)
# -----------------------------
# Генератор: примерный вход z
z_example = torch.randn(1, cfg.latent_dim, device=device)

# Дискриминатор: примерный вход image
x_example = torch.randn(1, cfg.channels, cfg.image_size, cfg.image_size, device=device)

try_make_torchviz_graph(G, z_example, "generator_graph", cfg.out_dir)
try_make_torchviz_graph(D, x_example, "discriminator_graph", cfg.out_dir)


# %%
# -----------------------------
# Обучение GAN: loss / optim
# -----------------------------
criterion = nn.BCEWithLogitsLoss()

opt_G = optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
opt_D = optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)

fixed_noise = torch.randn(cfg.fixed_noise_n, cfg.latent_dim, device=device)


# %%
# -----------------------------
# Шаги обучения
# -----------------------------
@torch.no_grad()
def sample_images(G: nn.Module, noise: torch.Tensor, out_path: str, nrow: int = 8) -> None:
    G.eval()
    fake = G(noise).detach().cpu()          # [-1,1]
    fake = (fake * 0.5 + 0.5).clamp(0, 1)   # -> [0,1] для сохранения
    grid = make_grid(fake, nrow=nrow)
    save_image(grid, out_path)
    print("saved:", out_path)


def train_discriminator(D: nn.Module, G: nn.Module, real: torch.Tensor) -> float:
    """
    Обучаем дискриминатор различать:
    - real: метка 1
    - fake (из G): метка 0
    Возвращаем значение loss_D.
    """
    D.train()
    G.eval()

    bsz = real.size(0)
    real = real.to(device, non_blocking=True)

    # real logits
    logits_real = D(real)
    y_real = torch.ones((bsz, 1), device=device)
    loss_real = criterion(logits_real, y_real)

    # fake logits (detach, чтобы не обновлять G)
    z = torch.randn(bsz, cfg.latent_dim, device=device)
    fake = G(z).detach()
    logits_fake = D(fake)
    y_fake = torch.zeros((bsz, 1), device=device)
    loss_fake = criterion(logits_fake, y_fake)

    loss_D = loss_real + loss_fake

    opt_D.zero_grad(set_to_none=True)
    loss_D.backward()
    opt_D.step()

    return float(loss_D.item())


def train_generator(D: nn.Module, G: nn.Module, bsz: int) -> float:
    """
    Обучаем генератор так, чтобы дискриминатор считал fake как real (метка 1).
    Возвращаем loss_G.
    """
    D.eval()
    G.train()

    z = torch.randn(bsz, cfg.latent_dim, device=device)
    fake = G(z)
    logits = D(fake)

    y = torch.ones((bsz, 1), device=device)  # хотим "обмануть" D
    loss_G = criterion(logits, y)

    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()
    opt_G.step()

    return float(loss_G.item())


# %%
# -----------------------------
# Основной цикл обучения
# -----------------------------
def train(cfg: Config) -> None:
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        lossD_mean = 0.0
        lossG_mean = 0.0
        n = 0

        for real, _ in train_loader:
            bsz = real.size(0)

            loss_D = train_discriminator(D, G, real)
            loss_G = train_generator(D, G, bsz)

            lossD_mean += loss_D
            lossG_mean += loss_G
            n += 1
            step += 1

        lossD_mean /= max(n, 1)
        lossG_mean /= max(n, 1)
        print(f"epoch {epoch:03d}/{cfg.epochs}: lossD={lossD_mean:.4f} lossG={lossG_mean:.4f}")

        # Сэмплы на фиксированном шуме
        out_path = os.path.join(cfg.out_dir, f"samples_epoch_{epoch:03d}.png")
        sample_images(G, fixed_noise, out_path, nrow=int(math.sqrt(cfg.fixed_noise_n)))


# %%
# В задании сказано, что запускать код НЕ нужно.
# Поэтому вызов обучения оставлен закомментированным:
# train(cfg)


# %% [markdown]
# # Пояснение: как работает этот код (GAN)
#
# **1) Данные.** Берём MNIST, приводим изображения к тензорам и нормируем в диапазон **[-1, 1]**. Это согласовано с выходной активацией генератора `tanh`, которая тоже выдаёт значения в [-1, 1].
#
# **2) Генератор (G).** На вход получает случайный вектор шума `z` размерности `latent_dim`. Далее:
# - линейной проекцией превращаем `z` в заготовку признаков размера `features*4 × 7 × 7`;
# - блоками `ConvTranspose2d` увеличиваем разрешение (7×7 → 14×14 → 28×28);
# - финальным `Conv2d + tanh` получаем изображение (1 канал, 28×28) в диапазоне [-1, 1].
#
# **3) Дискриминатор (D).** На вход получает изображение (реальное или сгенерированное), прогоняет через несколько свёрточных блоков с downsample и выдаёт **логит** (одно число). Сигмоиду явно не применяем, потому что используем `BCEWithLogitsLoss`, которая внутри сочетает sigmoid + BCE более численно устойчиво.
#
# **4) Обучение.**
# - Шаг D: хотим, чтобы D давал 1 на `real` и 0 на `fake`. Для `fake` используется `G(z).detach()`, чтобы градиенты не шли в генератор.
# - Шаг G: хотим “обмануть” D, то есть чтобы D давал 1 на `fake`. Поэтому метки для `fake` при обучении G — единицы.
#
# **5) Диаграммы моделей.** В блоке “Диаграммы моделей” создаём примерные входы (`z_example`, `x_example`) и строим вычислительный граф через `torchviz`. Результат сохраняется в файлы `generator_graph.png` и `discriminator_graph.png` (в папку `artifacts/`).
#
# **6) Визуальный контроль.** После каждой эпохи генерируем изображения на фиксированном шуме `fixed_noise` и сохраняем сетку в `samples_epoch_XXX.png`. Так можно видеть прогресс обучения без сравнения метрик.
#
# Примечание: запуск обучения в конце закомментирован, т.к. по условию его можно не запускать.