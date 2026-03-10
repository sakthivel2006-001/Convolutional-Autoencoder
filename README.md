# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Noise is a common issue in real-world image data, which affects performance in image analysis tasks. An autoencoder can be trained to remove noise from images, effectively learning compressed representations that help in reconstruction. The MNIST dataset (28x28 grayscale handwritten digits) will be used for this task. Gaussian noise will be added to simulate real-world noisy data.

## DESIGN STEPS

### STEP 1:
Load MNIST dataset and convert to tensors.

### STEP 2:
Apply Gaussian noise to images for training.

### STEP 3:
Design encoder-decoder architecture for reconstruction.

### STEP 4:
Use MSE loss to measure reconstruction quality.

### STEP 5:
Train autoencoder using Adam optimizer efficiently.

### STEP 6:
Evaluate model on noisy and clean images.

### STEP 7:
Visualize results comparing original, noisy, denoised versions.

### STEP 8:
Improve performance by tuning hyperparameters carefully.


## PROGRAM
### Name:SAKTHIVEL S
### Register Number: 212223220090


```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Define your layers here
        # Example:
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # For reconstruction, sigmoid is often used
        )
    def forward(self, x):
        # Include your code here
        x = x.view(-1, 28*28)  # Flatten the input image
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)  # Reshape to image dimensions
        return x

#Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
summary(model, (1, 28, 28))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
```

## OUTPUT

### Model Summary

<img width="1019" height="698" alt="Screenshot 2026-03-10 160342" src="https://github.com/user-attachments/assets/d28ebdfb-476b-40aa-9d0b-bad1649f9796" />



### Original vs Noisy Vs Reconstructed Image

<img width="1738" height="757" alt="Screenshot 2026-03-10 160456" src="https://github.com/user-attachments/assets/3bb8b31a-6084-4009-bb4f-7c4e6f9b8bb5" />




## RESULT

A convolutional autoencoder for image denoising application is developed successfully.
