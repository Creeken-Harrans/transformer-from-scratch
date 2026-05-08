from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 1.0,
        "seq_len": 350,
        "d_model": 512,
        "dataset_dir": "data/opus_books_en_it",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights/rtx5060_bs8_seq350",
        "model_basename": "tmodel_rtx5060_bs8_seq350_",
        "preload": None,
        "tokenizer_file": "data/tokenizer_{0}.json",
        "experiment_name": "runs/rtx5060_bs8_seq350",
        "save_best_only": True,
        "save_every": 5
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
