import torch

path = "/home/mraffael/martone_project/Organoid-Image-Classification-Using-Deep-Learning/models/MedicalNet/pretrain/resnet_18.pth"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
state_dict = state_dict['model'] if 'model' in state_dict else state_dict
print("Model loaded successfully.")


for k,v in state_dict.items():
    print(f"{k:<50}| {v.shape}")
