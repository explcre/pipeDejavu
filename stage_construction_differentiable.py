import jax
import jax.numpy as jnp

def gumbel_softmax_sample(logits, temperature):
    gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(logits.shape, dtype=logits.dtype)))
    return jax.nn.softmax((logits + gumbel_noise) / temperature)

def differentiable_argmin(logits, temperature):
    probabilities = gumbel_softmax_sample(logits, temperature)
    return jnp.argmax(probabilities, axis=-1)

def training_dp_impl_differentiable(num_layers, num_devices, num_microbatches, submesh_choices,
                num_autosharding_configs, compute_cost, max_n_succ_stages, temperature):
    """The core implementation of the DP algorithm with a differentiable search space."""
    f = np.full((num_layers + 1, num_layers + 1, num_devices + 1),
                np.inf,
                dtype=np.float32)
    f_stage_max = np.full((num_layers + 1, num_layers + 1, num_devices + 1),
                          0.0,
                          dtype=np.float32)
    f_argmin = np.full((num_layers + 1, num_layers + 1, num_devices + 1, 3),
                       -1,
                       dtype=np.int32)
    f[0, num_layers, 0] = 0

    for s in range(1, num_layers + 1):
        for i in range(num_layers - 1, -1, -1):
            for j in range(1, num_devices + 1):
                for k in range(num_layers, i, -1):
                    for m, submesh in enumerate(submesh_choices):
                        n_submesh_devices = np.prod(np.array(submesh))
                        if n_submesh_devices <= j:
                            for n_config in range(num_autosharding_configs):
                                if s - 1 <= max_n_succ_stages[i, k - 1, m, n_config]:
                                    stage_cost = compute_cost[i, k - 1, m, n_config]
                                    new_cost = f[s - 1, k, j - n_submesh_devices] + stage_cost
                                    if stage_cost <= max_stage_cost:
                                        # Apply Gumbel-Softmax trick to the cost
                                        new_cost_diff = f[s, i, j] - new_cost
                                        update_prob = differentiable_argmin(new_cost_diff, temperature)
                                        f[s, i, j] = f[s, i, j] * (1 - update_prob) + new_cost * update_prob
                                        f_stage_max[s, i, j] = max(f_stage_max[s - 1, k, j - n_submesh_devices], stage_cost)
                                        f_argmin[s, i, j] = differentiable_argmin(new_cost_diff, temperature)

    # ... the rest of the function remains unchanged
