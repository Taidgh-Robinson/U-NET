from huggingface_hub import login, upload_file
import glob
import os


class HuggingFaceHubClient:
    def __init__(self):
        login()
        self.repo_id = "taidgh-robinson/UNet-Whitepaper"
        self.repo_type = "model"

    def upload_model_to_repo(self, model_path, repo_path):
        upload_file(
            path_or_fileobj=model_path,
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )

    def upload_last_model_to_repo(self, model_directory, repo_path):
        files = glob.glob(os.path.join(model_directory, "*.pth"))
        files = [os.path.basename(file) for file in files]
        latest = max(files, key=lambda f: int(f.split("-")[0]))
        self.upload_last_model_to_repo(os.path.join(model_directory, latest), repo_path)
