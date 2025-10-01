from monai.networks.nets import SwinUNETR
from models.SwinUNETREncoder import SwinUNETREncoder

model = SwinUNETR(4,3)
print(model)
new_model = SwinUNETREncoder(model)
print(new_model)