import torch
import torch.nn as nn
import torch.optim as optim
from sympy import false
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import ssl
import matplotlib.pyplot as plt
import numpy as np

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

# Configure matplotlib for proper display
def setup_matplotlib():
    plt.rcParams['font.family'] = 'DejaVu Sans'

# Dataset loading function
def load_dataset(batch_size=64, subset_ratio=1.0):

    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create data directory
    os.makedirs('./data', exist_ok=True)

    # Load training and test sets
    train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

    # Use subset if specified
    if subset_ratio < 1.0:
        torch.manual_seed(42)
        train_size = int(len(train_dataset) * subset_ratio)
        test_size = int(len(test_dataset) * subset_ratio)
        train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset))[:train_size])
        test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset))[:test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

# Define neural network model
class HandwritingRecognitionModel(nn.Module):
    def __init__(self):
        super(HandwritingRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # Convolutional layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Convolutional layer 2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)   # Fully connected layer 1
        self.fc2 = nn.Linear(128, 10)  # Output layer (10 classes)

    def forward(self, x):
        # Forward pass implementation
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleEnsemble(nn.Module):
    def __init__(self, num_models=3):
        super(SimpleEnsemble, self).__init__()
        self.models = nn.ModuleList([HandwritingRecognitionModel() for _ in range(num_models)])

    def forward(self, x):
        return torch.mean(torch.stack([model(x) for model in self.models]), dim=0)

# Train and test model function
def train_and_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device='cpu'):

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    num_classes = 10  # KMNIST has 10 classes

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"Epoch {epoch + 1}/{num_epochs} - Training...")

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Convert labels to one-hot encoding (for MSE loss)
            labels_one_hot = torch.zeros(labels.size(0), num_classes, device=device)
            labels_one_hot.scatter_(1, labels.view(-1, 1), 1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels_one_hot)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        avg_train_acc = 100 * correct_train / total_train
        train_losses.append(avg_loss)
        train_accuracies.append(avg_train_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Train Acc: {avg_train_acc:.2f}%")


        # Testing phase
        model.eval()
        correct_test = 0
        total_test = 0

        print(f"Epoch {epoch + 1}/{num_epochs} - Testing...")

        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        avg_test_acc = 100 * correct_test / total_test
        test_accuracies.append(avg_test_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} - Test Acc: {avg_test_acc:.2f}%")
        print("-" * 50)

    return train_losses, train_accuracies, test_accuracies


# Visualize training results
def plot_metrics(results_dict, experiment_name, num_figures=1):

    # Check if it's a tuple (single result)
    if isinstance(results_dict, tuple) and len(results_dict) == 3:
        train_losses, train_accuracies, test_accuracies = results_dict
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, train_accuracies, 'g-', label='Training Accuracy')
        plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
        plt.legend()
        plt.title(f'{experiment_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.grid(True)
        plt.savefig(f'{experiment_name.replace(" ", "_")}.png')
        plt.show()
        return

    # Multiple parameter results
    param_names = list(results_dict.keys())
    params_per_figure = max(1, len(param_names) // num_figures)

    for fig_idx in range(num_figures):
        plt.figure(figsize=(10, 6))
        start_idx = fig_idx * params_per_figure
        end_idx = min(start_idx + params_per_figure, len(param_names))

        for param in param_names[start_idx:end_idx]:
            epochs = range(1, len(results_dict[param]['train_losses']) + 1)
            plt.plot(epochs, results_dict[param]['train_losses'], '-', label=f'{param} - Loss')
            plt.plot(epochs, results_dict[param]['train_accuracies'], '--', label=f'{param} - Train Acc')
            plt.plot(epochs, results_dict[param]['test_accuracies'], ':', label=f'{param} - Test Acc')

        plt.title(f'{experiment_name} ({fig_idx + 1}/{num_figures})')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{experiment_name.replace(" ", "_")}_figure{fig_idx + 1}.png')
        plt.show()


# Visualize prediction results
def visualize_predictions(model, test_loader, num_samples=100, device='cpu'):

    model.eval()

    # Get prediction results
    all_images = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_images.extend(images.cpu())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            if len(all_images) >= num_samples:
                break

    # Limit sample count
    all_images = all_images[:num_samples]
    all_labels = all_labels[:num_samples]
    all_preds = all_preds[:num_samples]

    # Create grid display
    rows = 10
    cols = 10
    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        img = all_images[i].squeeze().numpy() * 0.5 + 0.5
        plt.imshow(img, cmap='gray')
        color = 'green' if all_preds[i] == all_labels[i] else 'red'
        plt.title(f"P:{all_preds[i]}\nT:{all_labels[i]}", color=color, fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()


def run_experiment(model_class, train_loader, test_loader, experiment_type,
                           parameters, num_epochs=10, device='cpu'):

    results = {}

    for param_name, param_value in parameters.items():
        print(f"\nUsing {experiment_type}: {param_name}")
        model = model_class().to(device)

        # Set parameters based on experiment type
        if experiment_type == "Loss Function":
            criterion = param_value
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(),
                                   lr=param_value if experiment_type == "Learning Rate" else 0.001)

        # Train and test
        metrics = train_and_test(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

        # Store results
        results[param_name] = {
            'model': model,
            'train_losses': metrics[0],
            'train_accuracies': metrics[1],
            'test_accuracies': metrics[2]
        }

    # Plot charts
    num_figures = 2 if experiment_type in ["Learning Rate", "Batch Size"] else 1
    plot_metrics(results, f"{experiment_type} Comparison", num_figures=num_figures)

    return results


# Loss function experiment wrapper
def experiment_loss_functions(model_class, train_loader, test_loader, num_epochs=10, device='cpu'):
    loss_functions = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
        'MSELoss': nn.MSELoss()
    }
    results = run_experiment(model_class, train_loader, test_loader, "Loss Function",
                             loss_functions, num_epochs, device)

    return results


# Learning rate experiment wrapper
def experiment_learning_rates(model_class, train_loader, test_loader, num_epochs=10, device='cpu',
                             learning_rates=None):
    if learning_rates is None:
        learning_rates = {
            '0.1': 0.1,
            '0.01': 0.01,
            '0.001': 0.001,
            '0.0001': 0.0001
        }

    return run_experiment(model_class, train_loader, test_loader, "Learning Rate",
                          learning_rates, num_epochs, device)


# Batch size experiment wrapper
def experiment_batch_sizes(model_class, train_dataset, test_dataset, num_epochs=10, device='cpu',
                           batch_sizes=None, plot_individual=False):
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 64, 128]

    # Create batch size dictionary
    batch_sizes_dict = {str(bs): bs for bs in batch_sizes}

    # Create loaders for different batch sizes
    loaders_dict = {}
    for bs_name, bs_value in batch_sizes_dict.items():
        train_loader = DataLoader(train_dataset, batch_size=bs_value, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs_value, shuffle=False)
        loaders_dict[bs_name] = (train_loader, test_loader)

    # Run experiments
    results = {}
    for bs_name, loaders in loaders_dict.items():
        print(f"\nUsing Batch Size: {bs_name}")
        model = model_class().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train and test
        train_losses, train_accuracies, test_accuracies = train_and_test(
            model, loaders[0], loaders[1], criterion, optimizer, num_epochs, device
        )

        # Save results
        results[bs_name] = {
            'model': model,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }

        # Conditionally plot individual batch size results
        if plot_individual:
            plot_metrics((train_losses, train_accuracies, test_accuracies),
                         f"Batch Size={bs_name}")

    plot_metrics(results, "Batch Size Comparison", num_figures=2)

    return results


# Main function
if __name__ == "__main__":
    # Set up matplotlib
    setup_matplotlib()
    torch.manual_seed(42)
    np.random.seed(42)

    # Experiment control parameters
    FAST_MODE = True  # Enable fast mode
    experiment_epochs = 5 if FAST_MODE else 10  # Epochs for fast vs regular mode
    subset_ratio = 0.3 if FAST_MODE else 1.0    # Data subset ratio for fast mode

    # Device detection and data loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, Data subset: {subset_ratio * 100:.1f}%, Training epochs: {experiment_epochs}")

    # Load dataset
    train_loader, test_loader, train_dataset, test_dataset = load_dataset(
        batch_size=64, subset_ratio=subset_ratio)

    # Default model experiment
    print("\nExperiment 1: Model performance with default parameters")
    model = HandwritingRecognitionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    metrics = train_and_test(model, train_loader, test_loader, criterion, optimizer, experiment_epochs, device)
    plot_metrics(metrics, "Default Parameters")

    # Loss function experiment
    print("\nExperiment 2: Comparing different loss functions")
    loss_functions = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
        'MSELoss': nn.MSELoss()
    }
    loss_results = experiment_loss_functions(
        HandwritingRecognitionModel, train_loader, test_loader, experiment_epochs, device)

    # Learning rate experiment
    print("\nExperiment 3: Comparing different learning rates")
    learning_rates = {'0.01': 0.01, '0.001': 0.001} if FAST_MODE else {
        '0.1': 0.1, '0.01': 0.01, '0.001': 0.001, '0.0001': 0.0001
    }
    lr_results = experiment_learning_rates(
        HandwritingRecognitionModel, train_loader, test_loader, experiment_epochs, device, learning_rates)

    # Batch size experiment
    print("\nExperiment 4: Comparing different batch sizes")
    batch_sizes = [16, 64, 128] if FAST_MODE else [8, 16, 32, 64, 128]
    bs_results = experiment_batch_sizes(
        HandwritingRecognitionModel, train_dataset, test_dataset, experiment_epochs, device, batch_sizes)

    print("\nExperiment 5: Simple Ensemble Model")

    # Create Integration Models
    num_ensemble = 3  # Integration using 3 model compositions
    ensemble = SimpleEnsemble(num_ensemble).to(device)

    # Train each base model
    for i, model in enumerate(ensemble.models):
        print(f"\nTraining ensemble model {i + 1}/{num_ensemble}")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_and_test(model, train_loader, test_loader, criterion, optimizer, experiment_epochs, device)

    # Evaluate Integration Models
    ensemble.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = ensemble(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    ensemble_accuracy = 100 * correct / total
    print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.2f}%")
    print(f"Single Model Accuracy: {metrics[2][-1]:.2f}%")

    # Visualize model predictions
    visualize_predictions(model, test_loader, num_samples=100, device=device)

    print("\nAll experiments completed!")