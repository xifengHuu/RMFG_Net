from PIL import Image
import random
from torchvision.transforms import ToTensor as torchtotensor


################################### PRETRAIN #####################################################
class PretrainComposeTensor(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


class PretrainRandomHorizontalFlip(object):
    def _horizontal_flip(self, img, label):
        return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, label):
        if random.random() < self.prob:
            return self._horizontal_flip(img, label)
        else:
            return img, label


class PretrainRandomCropResize(object):
    def _randomCrop(self, img, label):
        width, height = img.size
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.NEAREST)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, label):
        return self._randomCrop(img, label)


class PretrainResize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, label):
        img = img.resize((self.width, self.height), Image.BILINEAR)
        label = label.resize((self.width, self.height), Image.NEAREST)
        return img, label


class PretrainNormalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img, label):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img, label


class PretrainToTensor(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, img, label):
        img, label = self.totensor(img), self.totensor(label).long()
        return img, label


################################### FINETUNE #####################################################
class FinetuneRandomCropResize(object):
    def _randomCrop(self, img, label, x, y):
        width, height = img.size
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.NEAREST)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        for img, label in zip(imgs, labels):
            img, label = self._randomCrop(img, label, x, y)
            res_img.append(img)
            res_label.append(label)
        return res_img, res_label


class FinetuneRandomHorizontalFlip(object):
    def _horizontal_flip(self, img, label):
        return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, imgs, labels):
        if random.random() < self.prob:
            res_img = []
            res_label = []
            for img, label in zip(imgs, labels):
                img, label = self._horizontal_flip(img, label)
                res_img.append(img)
                res_label.append(label)
            return res_img, res_label
        else:
            return imgs, labels


class FinetuneResize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        for img, label in zip(imgs, labels):
            res_img.append(img.resize((self.width, self.height), Image.BILINEAR))
            res_label.append(label.resize((self.width, self.height), Image.NEAREST))
        return res_img, res_label


class FinetuneNormalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, imgs, labels):
        res_img = []
        for img in imgs:
            for i in range(3):
                img[:, :, i] -= float(self.mean[i])
            for i in range(3):
                img[:, :, i] /= float(self.std[i])
            res_img.append(img)
        return res_img, labels


class FinetuneToTensor(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        for img, label in zip(imgs, labels):
            img, label = self.totensor(img), self.totensor(label).long()
            res_img.append(img)
            res_label.append(label)
        return res_img, res_label


################################### TEST #####################################################
class TestComposeTensor(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class TestResize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        img = img.resize((self.width, self.height), Image.BILINEAR)
        return img


class TestNormalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img


class TestToTensor(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, img):
        img = self.totensor(img)
        return img