import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import tqdm as tqdm
from torch.utils.data import DataLoader

from GAN import Discriminator, Generator, init_weights
from inception import inception_score



#Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

lr = 0.0002
batch_size = 128
image_size = 64
channels_img = 1
num_epochs = 200
channels_noise = 256
features_dis = 64
features_gen = 64
save_outputs = False # Plots output as images and saves them at intervals
save_interval = 100
load_current_model = True

#converting images to tensors
transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))]
)

#loading in the dataset
dataset = datasets.MNIST(root='../MNIST', train=True, transform=transforms, download=True)
train_set, test_set = torch.utils.data.random_split(dataset, [40000, 20000])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


#number or training sest images
a = len(dataset)
#print(a)

#showing the different labels for each category
b = dataset.train_labels
print(b)

#showing that the mnist dataset is a balanced dataset, as all classes have same number of instances
c = dataset.train_labels.bincount()
#print(c) --> tensor([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])

#Instantiating the networks
discriminator_net = Discriminator(channels_img, features_dis).to(device)
generator_net = Generator(channels_noise, channels_img, features_gen).to(device)

init_weights(discriminator_net)
init_weights(generator_net)


#optimizer
optimizer_dis = optim.Adam(discriminator_net.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_gen = optim.Adam(generator_net.parameters(), lr=lr, betas=(0.5, 0.999))


discriminator_net.train()
generator_net.train()

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

fixed_noise = torch.randn(64, channels_noise, 1, 1).to(device)

step = 0

print("training is starting...")

# Load previsously trained models from pth file
if load_current_model:
    discriminator_net.load_state_dict(torch.load('models/dis_model(600).pth'))
    generator_net.load_state_dict(torch.load('models/gen_model(600).pth'))

# Will contain avergae loss values for each epoch 
total_discriminator_loss = []
total_generator_loss = []
total_Dx = []

# Main training loop
for epoch in range(num_epochs):
    
    discriminator_loss_epoch = []
    generator_loss_epoch = []
    Dx_epoch = []
    
    for batch_idx, (data, _) in enumerate(tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        
        data = data.to(device)
        batch_size = data.shape[0]

        #Training the discriminator:
        discriminator_net.zero_grad()

        data.flatten()

        output = discriminator_net(data)

        label = (torch.ones_like(output)*0.9)

        lossD_real = criterion(output, label)
        D_x = output.mean().item()
        
        Dx_epoch.append(D_x)

        noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
        fake = generator_net(noise)
        label = (torch.ones_like(output) * 0.1).reshape(-1)

        output = discriminator_net(fake.detach()).reshape(-1)
        lossD_fake = criterion(output, label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizer_dis.step()
        
        lossD = lossD.detach().cpu().numpy()  
        discriminator_loss_epoch.append(lossD)

        #training Generator Network
        generator_net.zero_grad()
        output = discriminator_net(fake).reshape(-1)
        label = torch.ones_like(output)

        lossG = criterion(output, label)
        lossG.backward()
        optimizer_gen.step()
        
        lossG = lossG.detach().cpu().numpy() 
        generator_loss_epoch.append(lossG)
        
        if batch_idx % 100 == 0:
            step += 1
            #print(f"Epoch [{epoch+1}/{num_epochs}] Discriminator Loss: {lossD: .4f}, Generator Loss: {lossG: .4f} D(x): {D_x:-4f}")
            if save_outputs:
                with torch.no_grad():
                    idx = 0
                    fake = generator_net(fixed_noise)
                    img_grid_real = data[idx].cpu().numpy()
                    img_grid_fake = fake.detach()[idx].cpu().numpy()
                    plt.imshow(img_grid_real.transpose(1,2,0))
                    plt.imshow(img_grid_fake.transpose(1,2,0))
                    plt.savefig(f'gan_fake_test{epoch}_{batch_idx}.png')

    epoch_lossD = np.mean(discriminator_loss_epoch)
    epoch_lossG = np.mean(generator_loss_epoch)
    epoch_Dx = np.mean(Dx_epoch)
    
    total_discriminator_loss.append(epoch_lossD)
    total_generator_loss.append(epoch_lossG)
    total_Dx.append(epoch_Dx)
    
    #print(f"Epoch [{epoch+1}/{num_epochs}] Discriminator Loss: {epoch_lossD: .4f}, Generator Loss: {epoch_lossG: .4f} D(x): {epoch_Dx:-4f}")
    
    # Saves model after certian number fo epochs have passed
    if epoch % save_interval == 0 and epoch != 0:
        print('saving model')
        torch.save(discriminator_net.state_dict(), 'dis_model.pth')
        torch.save(generator_net.state_dict(), 'gen_model.pth')



#########################################################
# All temp code for saving, loading and showing outputs #
#########################################################

#'''
print(total_discriminator_loss)
print(total_generator_loss)
print(total_Dx)

losstemp1 = np.array(total_discriminator_loss)
losstemp2 = np.array(total_generator_loss)
losstemp3 = np.array(total_Dx)

from numpy import savetxt, loadtxt

savetxt('discriminator_loss(NEW200)', losstemp1, delimiter=',')
savetxt('generator_loss(NEW200)', losstemp2, delimiter=',')
savetxt('D_x(NEW200)', losstemp3, delimiter=',')

new1 = loadtxt('discriminator_loss(NEW200)')
new2 = loadtxt('generator_loss(NEW200)')
new3 = loadtxt('D_x(NEW200)')

print(new1)
print(new2)
print(new3)
#'''

# PLotting code used temporarily
import matplotlib.pyplot as plt

#'''
plt.figure(figsize=(12,8))
plt.plot(new1, label='Discriminator Loss')
plt.plot(new2, label='Generator Loss')
plt.plot(new3, label='D_x')
plt.xlabel('Epoch', fontsize=20)
#plt.ylabel('Loss', fontsize=20)
plt.hlines(0, 0, len(new1), color='r')
plt.legend(loc='lower right')
plt.grid()
#'''