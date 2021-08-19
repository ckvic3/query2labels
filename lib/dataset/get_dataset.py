import os

import torchvision.transforms as transforms
from lib.dataset.cocodataset import CoCoDataset


def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])

    train_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(brightness=0.3,
                                                                   contrast=0.3,
                                                                   saturation=0.3,
                                                                   hue=0),
                                            transforms.ToTensor(),
                                            normalize])


    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        val_dataset = CoCoDataset(
            image_dir=os.path.join(args.root, 'val2014'),
            anno_path=os.path.join(args.root, 'annotations2014/annotations','instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='./data/coco/val_label_vectors_coco14.npy',
        )
        train_dataset = CoCoDataset(
            image_dir=os.path.join(args.root, 'train2014'),
            anno_path=os.path.join(args.root, 'annotations2014/annotations','instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='./data/coco/train_label_vectors_coco14.npy'
        )

    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset


