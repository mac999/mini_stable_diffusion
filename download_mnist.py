import torch, torchvision, matplotlib.pyplot as plt
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

unique_images, unique_labels = next(iter(train_loader))
unique_images = unique_images.numpy()

row, column = 4, 16
fig, axes = plt.subplots(row, column, figsize=(16, 4), sharex=True, sharey=True)  

for i in range(row):  
    for j in range(column):  
        index = i * column + j  
        axes[i, j].imshow(unique_images[index].squeeze(), cmap='gray') 
plt.show()