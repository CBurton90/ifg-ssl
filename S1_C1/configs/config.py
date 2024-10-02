import toml
import munch

def load_global_config(filepath:str="project_config.toml"):
    return munch.munchify(toml.load(filepath))

def save_global_config(new_config, filepath:str="project_config.toml"):
    with open(filepath, "w") as file:
        toml.dump(new_config, file)