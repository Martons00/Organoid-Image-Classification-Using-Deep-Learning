import torch

path = "/home/mraffael/martone_project/Organoid-Image-Classification-Using-Deep-Learning/pretrained_models/BTCV/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
state_dict = state_dict['model'] if 'model' in state_dict else state_dict
print("Model loaded successfully.")

path2 = "/home/mraffael/martone_project/Organoid-Image-Classification-Using-Deep-Learning/pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/model_swinvit.pt"
checkpoint2 = torch.load(path2, map_location="cpu")
state_dict2 = checkpoint2.get("state_dict", checkpoint2)
print("Keys in the second state dict:")
for (k1,v1), (k2,v2) in zip(state_dict.items(), state_dict2.items()):
    print(f"{k1:<40}|  {k2}")
    print(f"{str(v1.shape):<40}|  {v2.shape}")

new_state_dict = {}
for k, v in state_dict2.items():
    if k.startswith('module.'):
        new_key = k[len('module.'):]
        new_key = new_key.replace('module.', '')
    else:
        new_key = k
    new_state_dict[new_key] = v
print("Keys in the new state dict:")

new_state_dict2 = {}
for k, v in state_dict.items():
    if k.startswith('swinViT.'):
        new_key = k[len('swinViT.'):]
        new_key = new_key.replace('linear', 'fc')
        new_key = new_key.replace('swinViT.', '')
    else:
        new_key = k
    new_state_dict2[new_key] = v
print("Keys in the new state dict:")




count = 0
for k1,k2 in zip(new_state_dict2, new_state_dict):
    print(f"{k1:<40}|  {k2}")
    if k1 == k2:
        count += 1
print(f"Number of matching keys: {count}/{len(state_dict.keys())}")

count = 0
count_shape = 0
for v1, v2 in zip(state_dict.values(), new_state_dict.values()):
    if torch.equal(v1, v2):
        count += 1
    if v1.shape == v2.shape:
        count_shape += 1
print(f"Number of matching values: {count}/{len(state_dict)}")
print(f"Number of matching shapes: {count_shape}/{len(state_dict)}")
