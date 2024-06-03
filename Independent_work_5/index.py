"""Module"""

# pylint: disable=E1101

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch


NUM_SAMPLES = 1000
NOISE = 0.1
RANDOM_STATE = 1

X, y = make_moons(n_samples=NUM_SAMPLES, noise=NOISE, random_state=RANDOM_STATE)
fig, ax = plt.subplots(1)

ax.scatter(x=X[:, 0], y=X[:, 1], c=y, s=2, cmap=plt.cm.RdYlBu)
ax.set_aspect("equal", "box")
plt.show()

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=60
)

print(len(X_train), len(X_test), len(y_train), len(y_test))


def set_seed(seed):
    """set_seed"""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class MoonModel(nn.Module):
    """MoonModel"""

    def __init__(self, n1, n2):
        super().__init__()

        self.l1 = nn.Linear(in_features=2, out_features=n1)
        self.l2 = nn.Linear(in_features=n1, out_features=n2)
        self.l3 = nn.Linear(in_features=n2, out_features=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        """forward"""

        return self.l3(self.tanh(self.l2(self.tanh(self.l1(x)))))


def accuracy_fn(y_true, y_pred):
    """accuracy_fn"""

    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = MoonModel(10, 6).to(DEVICE)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=MODEL.parameters(), lr=0.1)

EPOCHS = 200

X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)


def plot_decision_boundary(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
    """plot_decision_boundary"""
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))

    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """plot_predictions"""

    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})


for epoch in range(EPOCHS):
    MODEL.train()

    y_logits = MODEL(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    MODEL.eval()
    with torch.inference_mode():
        test_logits = MODEL(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
        )

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(MODEL, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(MODEL, X_test, y_test)
