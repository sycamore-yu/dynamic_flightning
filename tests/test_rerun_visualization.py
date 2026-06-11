import os
import jax
import jax.numpy as jnp
from flightning.envs.dynamic_avoidance_env import DynamicAvoidanceEnv
from flightning.visualization.rerun_dynamic_avoidance import RerunVizAdapter, HAS_RERUN

def test_rerun_visualizer_smoke():
    env = DynamicAvoidanceEnv()
    key = jax.random.PRNGKey(42)
    state, obs = env.reset(key)
    scan_image = jnp.zeros((1, 36, 6))

    rrd_path = "test_output.rrd"
    if os.path.exists(rrd_path):
        os.remove(rrd_path)

    # Initialize viz adapter
    viz = RerunVizAdapter(save_path=rrd_path)

    if HAS_RERUN:
        # Log state
        viz.log_state(state, scan_image, step_idx=0)
        # Verify the .rrd file is generated and is not empty
        assert os.path.exists(rrd_path)
        assert os.path.getsize(rrd_path) > 0
    else:
        # If Rerun is not installed, it should not write any file and should not crash
        assert not os.path.exists(rrd_path)

    # Clean up
    if os.path.exists(rrd_path):
        os.remove(rrd_path)
