from torchvision import transforms

P=0.8

# train_transforms = transforms.Compose([
#
#     transforms.RandomOrder([
#         transforms.RandomApply([
#     transforms.RandomRotation((-180,180))
#             ],p=P),
#     transforms.RandomHorizontalFlip(p=P),
#     transforms.RandomVerticalFlip(p=P),
#     ]),
#     transforms.RandomApply([
#             transforms.ColorJitter(
#                 brightness=0.2,
#                 contrast=0.2,
#                 saturation=0.2,
#                 hue=0.2
#             ),
#             transforms.RandomResizedCrop(224),
#     ],p=P),
#
#
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# train_transforms = transforms.Compose([
#     transforms.RandomOrder([
#     transforms.RandomRotation(360),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#         ]),
#         transforms.RandomApply([
#         transforms.RandomOrder([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomVerticalFlip(p=0.5),
#             transforms.RandomApply([
#                 transforms.RandomRotation(360),
#                                 ],p=0.5)
#
#         ]),
#         transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3, fill=255),
#         transforms.RandomApply([
#             transforms.ColorJitter(
#                 brightness=0.3,
#                 contrast=0.3,
#                 saturation=0.3,
#                 hue=0.3
#             ),
#             transforms.RandomChoice
#             ([
#                                 transforms.CenterCrop(280),
#                                 transforms.CenterCrop(180),
#                                 transforms.CenterCrop(200),
#                                 transforms.CenterCrop(160),
#                                 transforms.CenterCrop(140),
#                                 transforms.RandomResizedCrop(224),
#             ])
#         ]),
#         transforms.RandomGrayscale(p=0.1),
#             ],p=0.9),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])


train_transforms = transforms.Compose([
    transforms.RandomApply([
    #transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3, fill=255),
        transforms.RandomApply([
    transforms.ColorJitter(
                brightness=0.2,
                contrast=0.4,
                saturation=0.4,
                hue=0.4
            ),
    # transforms.RandomAffine(degrees=20, translate=None, scale=None,
    #     shear = 30, resample=False, fillcolor=0)
            ],p=0.5),
    #transforms.RandomGrayscale(p=0.1),
    transforms.RandomOrder([
    transforms.RandomRotation(360),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
        ]),
    transforms.RandomChoice([
    transforms.RandomCrop(250),
    transforms.RandomCrop(224),
    transforms.CenterCrop(200),
    transforms.CenterCrop(224),
    transforms.CenterCrop(250),
        ]),
    ],p=0.9),

    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 'train': transforms.Compose([
#     transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3, fill=255),
#     transforms.RandomChoice([transforms.CenterCrop(180),
#                              transforms.CenterCrop(160),
#                              transforms.CenterCrop(140),
#                              transforms.CenterCrop(120),
#                              transforms.Compose([transforms.CenterCrop(280),
#                                                  transforms.Grayscale(3),
#                                                  ]),
#                              transforms.Compose([transforms.CenterCrop(200),
#                                                  transforms.Grayscale(3),
#                                                  ]),
#                              ]),
#     transforms.Resize((224, 224)),
#     transforms.ColorJitter(hue=(0.1, 0.2)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ]),
# 'valid': transforms.Compose([
#     transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3, fill=255),
#     transforms.RandomChoice([transforms.CenterCrop(180),
#                              transforms.CenterCrop(160),
#                              transforms.CenterCrop(140),
#                              transforms.CenterCrop(120),
#                              transforms.Compose([transforms.CenterCrop(280),
#                                                  transforms.Grayscale(3),
#                                                  ]),
#                              transforms.Compose([transforms.CenterCrop(200),
#                                                  transforms.Grayscale(3),
#                                                  ]),
#                              ]),
#     transforms.Resize((224, 224)),
#     transforms.ColorJitter(hue=(0.1, 0.2)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ]), }


# train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# train_datagen=ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         vertical_flip = True
#         )
