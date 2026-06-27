from torchvision import transforms


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

target_size = (256, 256)  # Target image size (because NN input has a fixed size dimension)

# ImageNet normalizer ( You can later replace this with the datasent mean and std)
imagenet_normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

# Basic transformation that resize the input into target_size
transform_basic = transforms.Compose([
                # transforms.Resize(size=(int(target_size[0]), int(target_size[1]))),
                transforms.ToTensor(),
                imagenet_normalize
            ])

def get_transform(target_size, transform_phase='train'):
    """
    Predefined transformation pipe for the dataset
    :param target_size: tuple of (W,H) result image from the pipe
    :param transform_phase: basic/train/val/test phase of different transformation e.g. test will not need RandomCrop
    :return: a transformation function to target_size
    """
    # check target_size
    if type(target_size) is int:
        target_size = (target_size, target_size)
    assert type(target_size) is tuple, "target_size must be tuple of (W:int, H:int) or int if square is needed"

    # enlarge 10% bigger for the later cropping
    enlarge = transforms.Resize(size=(int(target_size[0] * 1.1), int(target_size[1] * 1.1)))

    # ImageNet normalizer
    imagenet_normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    # Compose
    transform_dict = {
        'basic':
          transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                imagenet_normalize
            ]),
        # REWRITTEN 'train' phase using your safe brain-MRI pattern
        'train':
            transforms.Compose([
                transforms.Resize(target_size),
                transforms.RandomAffine(
                    degrees=8,               # Safe minor rotation
                    translate=(0.04, 0.04),  # Safe minor shifting
                    scale=(0.96, 1.04)       # Safe minor scaling
                ),
                transforms.ColorJitter(brightness=0.1, contrast=0.1), # Safe intensity tweaks
                transforms.ToTensor(),
                imagenet_normalize
            ]),
        'val':
            transforms.Compose([
                enlarge,
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                imagenet_normalize
            ]),
        'test':
            transforms.Compose([
                enlarge,
                # transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                imagenet_normalize
            ])
    }

    # check phase
    if transform_phase in transform_dict:
        return transform_dict[transform_phase]
    else:
        raise Exception("Unknown phase specified")


def get_transform_center_crop(target_size, scaling_factor=1.0):
  """
  Produce the centercropimage with the specific target_size.
  """
  return transforms.Compose([
      transforms.Resize(size=(int(target_size[0] * scaling_factor), int(target_size[1] * scaling_factor))),
      transforms.CenterCrop(target_size),
      transforms.ToTensor(),
      imagenet_normalize
  ])


# Define the dictionary of transform functions
transform_fn = {
    # 'basic': transform_basic,
    'basic': get_transform(target_size=target_size, phase='basic'),
    'center_crop': get_transform_center_crop(target_size=target_size, scaling_factor=1.3),
    'train': get_transform(target_size=target_size),
    'val': get_transform(target_size=target_size, phase='val'),
    'test': get_transform(target_size=target_size, phase='test'),
}
