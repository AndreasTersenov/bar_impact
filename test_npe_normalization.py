"""Test NPE normalization behavior."""
import numpy as np
import jax.numpy as jnp
import jax.random as random
from jaxili.inference import NPE

# Create synthetic data with known parameter ranges
np.random.seed(42)
n_sims = 1000
n_features = 10
n_params = 2

# Parameters in range [0, 1]
params = np.random.uniform(0, 1, size=(n_sims, n_params))
# Data with larger range [0, 100]
data = np.random.uniform(0, 100, size=(n_sims, n_features))

print("Training data ranges:")
print(f"  Params: min={params.min():.4f}, max={params.max():.4f}, mean={params.mean():.4f}")
print(f"  Data: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")

# Train NPE
npe = NPE()
npe = npe.append_simulations(jnp.array(params), jnp.array(data))

print("\nTraining NPE...")
metrics, _ = npe.train(
    checkpoint_path="/tmp/test_npe",
    num_epochs=50,
    learning_rate=0.001,
    training_batch_size=32,
)

# Build posterior
posterior = npe.build_posterior()

# Sample from posterior using test data
test_obs = data[0]  # Use first observation
print(f"\nTest observation: min={test_obs.min():.4f}, max={test_obs.max():.4f}")

key = random.PRNGKey(42)
samples = posterior.sample(x=test_obs, num_samples=100, key=key)
samples_np = np.array(samples)

print(f"\nPosterior samples:")
print(f"  Shape: {samples_np.shape}")
print(f"  Min: {samples_np.min(axis=0)}")
print(f"  Max: {samples_np.max(axis=0)}")
print(f"  Mean: {samples_np.mean(axis=0)}")
print(f"  True value: {params[0]}")

print(f"\nSamples in expected range? {np.all((samples_np >= 0) & (samples_np <= 1))}")
