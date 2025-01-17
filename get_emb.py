import torch
import sys
from pointnet.source.model import PointNet
import numpy

cloud1 = numpy.load("/home/h-okano/DiffBinaural/processed_data/pointcloud/000001.mp4/000002.npz")

cloud2 = numpy.load("/home/h-okano/DiffBinaural/processed_data/pointcloud/000001.mp4/000004.npz")

cloud1 = cloud1['depth_map_3d'].reshape(-1,3)
cloud2 = cloud2['depth_map_3d'].reshape(-1, 3)


cloud1 = torch.FloatTensor(cloud1)
cloud2 = torch.FloatTensor(cloud2)
cloud1 = cloud1.permute(1, 0)
cloud2 = cloud2.permute(1, 0)

clouds = torch.stack((cloud1, cloud2), dim=0)

model = PointNet()
model.load_state_dict(torch.load("/home/h-okano/DiffBinaural/pointnet/pretrained_model/save.pth"))
output = model.get_emb(clouds) 
print(output.shape)
print(output)