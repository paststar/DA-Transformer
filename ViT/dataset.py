import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as torchdata


def get_dataset(dataset_name, path='/database'):
    if dataset_name in ['amazon', 'dslr', 'webcam']:  # OFFICE-31
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        tr_dataset = datasets.ImageFolder(path + '/office31/' + dataset_name + '/images/', data_transforms['train'])
        te_dataset = datasets.ImageFolder(path + '/office31/' + dataset_name + '/images/', data_transforms['test'])
        print('{} train set size: {}'.format(dataset_name, len(tr_dataset)))
        print('{} test set size: {}'.format(dataset_name, len(te_dataset)))

    else:
        raise ValueError('Dataset %s not found!' % dataset_name)
    return tr_dataset, te_dataset

if __name__ == "__main__":
    batch_size = 16
    workers = 4
    source = 'dslr'
    target = 'amazon'
    path = '/home/bang/Desktop/Domain Adaptation/Dataset'
    src_trainset, src_testset = get_dataset(source, path = path)
    tgt_trainset, tgt_testset = get_dataset(target, path = path)
    src_train_loader = torchdata.DataLoader(src_trainset, batch_size=batch_size, shuffle=True,
                                            num_workers=workers, pin_memory=True, drop_last=True)
    tgt_train_loader = torchdata.DataLoader(tgt_trainset, batch_size=batch_size, shuffle=True,
                                            num_workers=workers, pin_memory=True, drop_last=True)
    tgt_test_loader = torchdata.DataLoader(tgt_testset, batch_size=batch_size, shuffle=True,
                                           num_workers=workers, pin_memory=True, drop_last=False)

    L = torch.tensor([0])
    for step, src_data in enumerate(tgt_train_loader):
        tgt_imgs, tgt_labels = src_data
        L=torch.cat((L, tgt_labels))
    print(torch.unique(L))


