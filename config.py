from pathlib import Path

class Config:
    def __init__(self):
        self.get_hf_token()

        self.batch_size = 64   # was 8
        self.num_epochs = 20
        self.lr = 1.4 * 2 * 10**-4 # was 1e-4
        self.seq_len = 70
        self.d_model = 512
        self.model_basename = "tmodel_"
        self.preload = "latest"
        self.experiment_name = "runs/tmodel"

        if True:
            self.datasource = "wmt/wmt14"
            self.datasource_name = "de-en"
            self.data_files =None
            self.model_folder = "wmt-wmt14"
        else:
            self.datasource = "Helsinki-NLP/opus-100"
            self.datasource_name = "de-en"
            self.data_files = None
            self.model_folder = "opus-100-de-en"

        self.lang_src = "de"
        self.lang_tgt = "en"

        self.tokenizer_file = self.model_folder + "/tokenizer_{0}.json"

    def get_hf_token(self):
        with open('hf_token.txt', 'r') as f:
            self.hf_token = f.read().strip()

    def get_weights_file_path(self, epoch: str):
        model_folder = self.model_folder
        model_filename = f"{self.model_basename}{epoch}.pt"
        return str(Path('.') / model_folder / model_filename)

    # Find the latest weights file in the weights folder
    def latest_weights_file_path(self):
        model_folder = self.model_folder
        model_filename = f"{self.model_basename}*"
        weights_files = list(Path(model_folder).glob(model_filename))
        if len(weights_files) == 0:
            return None
        weights_files.sort()
        return str(weights_files[-1])
