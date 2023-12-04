import torch


class AdversarialAttack:
    def __init__(self, model, loss_fn) -> None:
        self.model = model
        self.loss_fn = loss_fn

    def generate_adversarial_examples(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def save_adversarial_examples(self, X: torch.Tensor, y: torch.Tensor, filename: str) -> None:
        X_adv = self.generate_adversarial_examples(X, y)
        torch.save({"model_state_dict": self.model.state_dict(),
                    "examples": X_adv,
                    "labels": y},
                   f"{filename}")
        print(f"saved model and adversarial examples to {filename}")

    def evaluate_attack(self, X: torch.Tensor, y: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], float]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    # TODO add some other utility functions
