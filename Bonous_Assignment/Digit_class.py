import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import os

# ---------------------------
#  Hyperparameters
# ---------------------------
batch_size = 128
lr = 2e-4
num_epochs = 30
noise_dim = 100
embed_dim = 50
img_shape = (1, 28, 28)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "cgan_outputs"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
#  Data Loader
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True, drop_last=True)

# ---------------------------
#  Generator
# ---------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # noise: (B, noise_dim), labels: (B,)
        lbl = self.label_emb(labels)                # (B, embed_dim)
        x = torch.cat([noise, lbl], dim=1)          # (B, noise_dim+embed_dim)
        img = self.net(x)
        return img.view(-1, *img_shape)            # (B, 1, 28, 28)

# ---------------------------
#  Discriminator
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))) + embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # img: (B, 1, 28, 28), labels: (B,)
        img_flat = img.view(img.size(0), -1)         # (B, 784)
        lbl = self.label_emb(labels)                 # (B, embed_dim)
        x = torch.cat([img_flat, lbl], dim=1)        # (B, 784+embed_dim)
        return self.net(x)

# ---------------------------
#  Initialize models + optimizers + loss
# ---------------------------
G = Generator().to(device)
D = Discriminator().to(device)
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# ---------------------------
#  Training Loop
# ---------------------------
fixed_noise = torch.randn(10, noise_dim, device=device)
fixed_labels = torch.arange(0, 10, device=device)  # 0–9

G_losses, D_losses = [], []

for epoch in range(1, num_epochs + 1):
    for real_imgs, real_labels in loader:
        bs = real_imgs.size(0)
        real_imgs, real_labels = real_imgs.to(device), real_labels.to(device)

        # 1) Train Discriminator
        opt_D.zero_grad()
        # real
        valid = torch.ones(bs, 1, device=device)
        out_real = D(real_imgs, real_labels)
        loss_real = criterion(out_real, valid)
        # fake
        noise = torch.randn(bs, noise_dim, device=device)
        fake_labels = torch.randint(0, 10, (bs,), device=device)
        fake_imgs = G(noise, fake_labels)
        fake = torch.zeros(bs, 1, device=device)
        out_fake = D(fake_imgs.detach(), fake_labels)
        loss_fake = criterion(out_fake, fake)
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        opt_D.step()

        # 2) Train Generator
        opt_G.zero_grad()
        # aim to fool D
        out = D(fake_imgs, fake_labels)
        loss_G = criterion(out, valid)
        loss_G.backward()
        opt_G.step()

    G_losses.append(loss_G.item())
    D_losses.append(loss_D.item())

    # Generate sample row after each epoch
    with torch.no_grad():
        gen_imgs = G(fixed_noise, fixed_labels).cpu()
        grid = make_grid(gen_imgs, nrow=10, normalize=True, value_range=(-1, 1))
        save_image(grid, f"{output_dir}/epoch_{epoch:03d}.png")

    print(f"Epoch [{epoch}/{num_epochs}]  D_loss: {loss_D:.4f}  G_loss: {loss_G:.4f}")

# ---------------------------
#  Plot Loss Curves
# ---------------------------
plt.figure(figsize=(8,4))
plt.plot(G_losses, label="G loss")
plt.plot(D_losses, label="D loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/loss_curve.png")
plt.show()

# ---------------------------
#  Final Generation Grid
# ---------------------------
with torch.no_grad():
    final_imgs = G(fixed_noise, fixed_labels).cpu()
    final_grid = make_grid(final_imgs, nrow=10, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(10,2))
    plt.axis("off")
    plt.imshow(final_grid.permute(1,2,0), cmap="gray")
    plt.title("Generated MNIST digits 0–9")
    plt.show()
