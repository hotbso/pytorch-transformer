from pathlib import Path

def get_config():
    return {
        "batch_size": 16,   # was 8
        "num_epochs": 20,
        "lr": 1.4 * 10**-4, # was 1e-4
        "seq_len": 70,
        "d_model": 512,
        "datasource": "Yujivus/wmt14-de-en-50k-no-hospital",
        "datasource_base": "wmt14-de-en-50k-no-hospital",
         #"datasource": 'opus_books',
         #"datasource_base": 'opus_books',
        "lang_src": "de",
        "lang_tgt": "en",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource_base']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource_base']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
