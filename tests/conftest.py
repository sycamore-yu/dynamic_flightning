import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax


jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get("FLIGHTNING_JAX_CACHE_DIR", "/tmp/flightning_jax_cache"),
)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)


def pytest_configure(config):
    pluginmanager = config.pluginmanager
    for plugin_name in ("launch_ros", "launch_testing"):
        plugin = pluginmanager.get_plugin(plugin_name)
        if plugin is not None:
            pluginmanager.unregister(plugin, name=plugin_name)
