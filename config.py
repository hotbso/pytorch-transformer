from pathlib import Path

class Config:
    def __init__(self):
        self.batch_size = 16   # was 8
        self.num_epochs = 20
        self.lr = 1.4 * 10**-4 # was 1e-4
        self.seq_len = 70
        self.d_model = 512
        self.datasource = "Yujivus/wmt14-de-en-50k-no-hospital"
        self.datasource_base = "wmt14-de-en-50k-no-hospital"
        self.datasource_name = None
         #self.datasource = 'opus_books'
         #self.datasource_base = 'opus_books'
        self.lang_src = "de"
        self.lang_tgt = "en"
        self.model_folder = "weights"
        self.model_basename = "tmodel_"
        self.preload = "latest"
        self.tokenizer_file = "tokenizer_{0}.json"
        self.experiment_name = "runs/tmodel"

    def get_weights_file_path(self, epoch: str):
        model_folder = f"{self.datasource_base}_{self.model_folder}"
        model_filename = f"{self.model_basename}{epoch}.pt"
        return str(Path('.') / model_folder / model_filename)

    # Find the latest weights file in the weights folder
    def latest_weights_file_path(self):
        model_folder = f"{self.datasource_base}_{self.model_folder}"
        model_filename = f"{self.model_basename}*"
        weights_files = list(Path(model_folder).glob(model_filename))
        if len(weights_files) == 0:
            return None
        weights_files.sort()
        return str(weights_files[-1])
