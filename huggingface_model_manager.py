from huggingface_hub import login, upload_file

login()
upload_file(
    path_or_fileobj="model_state_dicts/full_model_adam/24-adam-policy_net.pth",
    path_in_repo="24-adam-policy_net.pth",
    repo_id="taidgh-robinson/UNet-Whitepaper",
    repo_type="model"
)