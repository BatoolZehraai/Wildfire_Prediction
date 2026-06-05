import matplotlib.pyplot as plt
import os

# Create docs folder if it doesn't exist
if not os.path.exists('docs'):
    os.makedirs('docs')
    print("📁 Created 'docs' folder.")

# Training loss values from your successful run
epochs = list(range(1, 21))
loss_values = [
    0.052989, 0.031138, 0.025240, 0.017648, 0.011300, 
    0.006976, 0.005019, 0.004048, 0.003484, 0.003139,
    0.002914, 0.002728, 0.002562, 0.002465, 0.002331,
    0.002290, 0.002136, 0.002086, 0.002043, 0.001961
]

plt.figure(figsize=(10, 5))
plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b', label='Training Loss')
plt.title('Wildfire Diffusion Model - Training Convergence')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# Save the plot
save_path = 'docs/training_loss_plot.png'
plt.savefig(save_path)
print(f"📈 Loss plot successfully saved in: {save_path}")
plt.show()