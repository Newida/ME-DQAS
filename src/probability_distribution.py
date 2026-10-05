from abc import ABC, abstractmethod
import numpy as np


class ProbabilityDistribution(ABC):
    @abstractmethod
    def sample(self, batch_size: int = 1):
        pass

    @abstractmethod
    def grad_log_prob(self, samples):
        pass

    @abstractmethod
    def reinforce_gradient(self, samples, values, baseline):
        pass

    @abstractmethod
    def update(self, g_phi, lr) -> None:
        pass


class FactorizedCategoricalProbabilityDistribution(ProbabilityDistribution):
    # Note: Hardware constraints might be added using masking
    # For now, we assume all choices are valid for all positions
    @staticmethod
    def _make_rng(seed: int | None) -> np.random.Generator:
        if seed is None:
            return np.random.default_rng()
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be a non-negative integer or None.")
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be a non-negative integer or None.")
        return np.random.Generator(np.random.PCG64(seed))

    def __init__(
        self,
        num_positions: int,
        num_choices: int,
        seed: int | None = None
    ):
        if num_positions <= 0:
            raise ValueError("num_positions must be positive.")
        if num_choices <= 0:
            raise ValueError("num_choices must be positive.")

        self.num_positions = num_positions
        self.num_choices = num_choices
        self.rng = self._make_rng(seed)
        self.phi = self.rng.normal(
            loc=0.0,
            scale=1e-8,
            size=(num_positions, num_choices),
        )

    @property
    def probabilities(self):
        shifted = self.phi - np.max(self.phi, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def most_likely(self):
        return np.argmax(self.probabilities, axis=1).astype(np.intp, copy=False)

    def sample(self, batch_size: int = 1):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        cdf = np.cumsum(self.probabilities, axis=1)
        draws = self.rng.random((batch_size, self.num_positions, 1))
        return np.sum(draws > cdf[None, :, :], axis=2).astype(np.intp, copy=False)

    def grad_log_prob(self, samples):
        samples = np.asarray(samples, dtype=np.intp)
        single_sample = samples.ndim == 1
        if single_sample:
            samples = samples[None, :]
        if samples.shape[1] != self.num_positions:
            raise ValueError("samples must have shape (batch_size, num_positions).")

        batch_size = samples.shape[0]
        scores = -np.broadcast_to(
            self.probabilities,
            (batch_size, self.num_positions, self.num_choices),
        ).copy()
        scores[
            np.arange(batch_size)[:, None],
            np.arange(self.num_positions)[None, :],
            samples,
        ] += 1.0
        return scores[0] if single_sample else scores

    def reinforce_gradient(self, samples, values, baseline):
        samples = np.asarray(samples, dtype=np.intp)
        if samples.ndim == 1:
            samples = samples[None, :]
        if samples.shape[1] != self.num_positions:
            raise ValueError("samples must have shape (batch_size, num_positions).")

        values = np.asarray(values, dtype=np.float64)
        if values.shape != (samples.shape[0],):
            raise ValueError("values must have shape (batch_size,).")

        if baseline is None:
            baseline_value = float(np.mean(values))
        elif hasattr(baseline, "value"):
            baseline_value = getattr(baseline, "value")
            if baseline_value is None:
                baseline_value = float(np.mean(values))
            else:
                baseline_value = float(baseline_value)
        else:
            baseline_value = float(baseline)
        advantages = values - baseline_value

        batch_size = samples.shape[0]
        grad = -np.mean(advantages) * self.probabilities
        for position in range(self.num_positions):
            grad[position] += np.bincount(
                samples[:, position],
                weights=advantages,
                minlength=self.num_choices,
            )[: self.num_choices] / batch_size
        return grad

    def update(self, g_phi, lr) -> None:
        self.phi -= lr * np.asarray(g_phi, dtype=np.float64)
