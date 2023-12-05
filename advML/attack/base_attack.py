import torch


class BaseAttack:
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
        X_adv = self.generate_adversarial_examples(X, y)

        self.model.eval()
        self.model.zero_grad()
        output = self.model(X_adv).detach()

        softmax = torch.nn.Softmax(-1)
        prediction = softmax(output)
        confidence, predicted_label = prediction.max(1)
        accuracy = (predicted_label == y).type(torch.float).sum().item()

        print(f"accuracy on adversarial examples: {accuracy}")

        return X_adv, (predicted_label, confidence)

    def __str__(self) -> str:
        raise NotImplementedError

    # TODO add some other utility functions -- like visualizing examples?
