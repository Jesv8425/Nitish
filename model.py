from huggingface_hub import snapshot_download

repo_id = "ultralytics/yolov8"
local_dir = "models"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print(f"Model repo downloaded to: {local_dir}")

from transformers import AutoModel

# Check if a YOLOv8 .hf model exists (replace with actual repo)
# model = AutoModel.from_pretrained("username/yolov8-hf-format")
# model.save_pretrained("yolov8_hf")