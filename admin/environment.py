import importlib


def env_settings():
    env_module_name = 'admin.local'
    env_module = importlib.import_module(env_module_name)
    return env_module.EnvironmentSettings()
