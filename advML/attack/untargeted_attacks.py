import torch
from .adversarial_attack import AdversarialAttack


class FGSMAttack(AdversarialAttack):
    def __init__(self, model, loss_fn, epsilon: int | float) -> None:
        super().__init__(model, loss_fn)
        self.epsilon = epsilon

    def generate_adversarial_examples(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        X = X.detach().clone()
        X.requires_grad = True

        self.model.eval()
        self.model.zero_grad()
        output = self.model(X)
        loss = self.loss_fn(output, y)
        loss.backward()
        eta = self.epsilon * X.grad.detach().sign()
        X_adv = (X + eta).clamp(0, 1)
        return X_adv

    def save_adversarial_examples(self, X: torch.Tensor, y: torch.Tensor, filename: str) -> None:
        return super().save_adversarial_examples(X, y, filename)

    def evaluate_attack(self, X: torch.Tensor, y: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], float]:
        X_adv = self.generate_adversarial_examples(X, y)

        self.model.eval()
        output = self.model(X_adv)

        softmax = torch.nn.Softmax(-1)
        prediction = softmax(output)
        predicted_label = prediction.argmax(1)
        accuracy = (predicted_label == y).type(torch.float).sum().item()

        print("accuracy on adversarial examples (epsilon={}): {}"
              .format(self.epsilon, accuracy))

        return (X_adv, predicted_label), accuracy

    def __str__(self) -> str:
        return f"FGSMAttack(epsilon={self.epsilon})"


class PGDAttack(AdversarialAttack):
    def __init__(self, model, loss_fn, epsilon: float | int, alpha: float | int, num_iters: int, random_start: bool = False) -> None:
        super().__init__(model, loss_fn)
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iters = num_iters
        self.random_start = random_start

    def generate_adversarial_examples(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        X = X.detach().clone()
        X_adv = X.detach().clone()

        if self.random_start:
            X_adv += torch.FloatTensor(1).uniform_(-self.epsilon, self.epsilon)
            X_adv = X_adv.clamp(0, 1)

        self.model.eval()
        for _ in range(self.num_iters):
            self.model.zero_grad()
            X_adv.requires_grad = True
            output = self.model(X_adv)
            loss = self.loss_fn(output, y)
            loss.backward()
            eta = self.alpha * X_adv.grad.detach().sign()
            X_adv = (X_adv.detach() + eta).clamp(X - self.epsilon, X + self.epsilon)
            X_adv = X_adv.clamp(0, 1)

        return X_adv

    def evaluate_attack(self, X: torch.Tensor, y: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], float]:
        X_adv = self.generate_adversarial_examples(X, y)

        self.model.eval()
        output = self.model(X_adv)

        softmax = torch.nn.Softmax(-1)
        prediction = softmax(output)
        predicted_label = prediction.argmax(1)
        accuracy = (predicted_label == y).type(torch.float).sum().item()

        print("accuracy on adversarial examples (epsilon={}): {}"
              .format(self.epsilon, accuracy))

        return (X_adv, predicted_label), accuracy

    def __str__(self) -> str:
        return f"PGDAttack(epsilon={self.epsilon}, alpha={self.alpha}, num_iters={self.num_iters})"
