import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
# from torchvision
from PIL import Image
from skimage import transform
import skimage.io as io
from pathlib import PurePath

class CelebADataset(Dataset):

    def __init__(self, root, image_size, split):
        self.root = root
        self.image_size = image_size
        self.split = split
        self.split_code_map = {
            "train": "0",
            "eval": "1",
            "test": "2"
        }
        self.file_list = self.read_file_list()

    
    def __len__(self):
        return len(self.file_list)

    
    def __getitem__(self, idx):
        original_image = np.float64(io.imread(self.file_list[idx]))
        original_image = transform.resize(original_image, self.image_size)
        original_image = original_image.reshape((3,)+self.image_size)
        original_image = (original_image-np.mean(original_image))/np.max(np.abs(original_image))

        # Patch
        mask = np.ones(self.image_size,dtype=np.float32)
        x = np.random.randint(self.image_size[0]//6,5*self.image_size[0]//6)
        y = np.random.randint(self.image_size[1]//6,5*self.image_size[1]//6)
        h = np.random.randint(self.image_size[0]//4,self.image_size[0]//2)
        w = np.random.randint(self.image_size[1]//4,self.image_size[1]//2)
        mask[max(0,x-h//2):min(self.image_size[0],x+h//2),max(0,y-w//2):min(self.image_size[1],y+w//2)] = 0
        target_image = original_image.copy()
        target_image[0][1-mask > 0.5] = np.max(target_image)

        return torch.FloatTensor(target_image), torch.FloatTensor(original_image), torch.FloatTensor(mask)
    

    def read_file_list(self):
        root_path = PurePath(self.root)
        eval_file = root_path.joinpath("list_eval_partition.txt")
        file_list = list()
        with open(eval_file, 'r') as f:
            line = f.readline()
            while line:
                space_split = line.strip().split(" ")
                # print(space_split)
                if self.split_code_map[self.split] == space_split[1]:
                    filename = space_split[0].split(".")[0]
                    filename = filename + ".png"
                    file_list.append(str(root_path.joinpath("img_align_celeba_png", filename)))
                line = f.readline()
        return file_list

    


if __name__ == "__main__":
    dset = CelebADataset("/home/ssing57/dataset", (64, 64), "train")
    # print(dset.read_file_list())

    # train_celeba_data = torchvision.datasets.ImageFolder(root='/home/ssing57/dataset', transform=torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(IMAGE_SIZE),
    #     torchvision.transforms.CenterCrop(IMAGE_SIZE),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ]))
    train_dataloader = torch.utils.data.DataLoader(dset, batch_size=4, shuffle=True)

    # # Decide which device we want to run on
    # # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # # Plot some training images
    real_batch = next(iter(train_dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    print(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu().shape)
    a = np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)).numpy()
    # print(a)
    im = Image.fromarray((a * 255).astype(np.uint8))
    im.save("file.jpeg")
    # plt.savefig()
