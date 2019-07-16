import cv2 as cv
import numpy as np
import random


class Mask:
    def __init__(self, img, label, mask_addr, mask_cls):
        self.img = img
        self.label = label
        self.mask_addr = mask_addr
        self.mask_cls = mask_cls

    def get_mask(self, mask_number=3, mid_size_control=0):
        label_new = self.label
        height, width = np.shape(self.img)[:2]
        with open(self.mask_addr, 'r') as f:
            addrs = f.read().splitlines()
        len_mask = len(addrs)
        for i in range(mask_number):
            mask = cv.imread(random.randint(0, len_mask - 1))
            if mid_size_control != 0:
                mh, mw = np.shape(mask)[:2]
                mask = cv.resize(mask, (int(mw * height / (10 * mh)), height // 10))
            mh, mw = np.shape(mask)[:2]
            loc_good = True
            loc_x = 0
            loc_y = 0
            while loc_good:
                loc_y = random.randint(mh, height - 1)
                loc_x = random.randint(mw, width - 1)
                if self.loc_suit(mask, loc_x, loc_y, height, width, label_new):
                    loc_good = False
            xc, yc, w, h = [(loc_x - mw / 2) / width, (loc_y - mh / 2) / height, mw / width, mh / height]
            np.append(label_new, np.array([self.mask_cls, xc, yc, w, h]))
            self.img[loc_x - mw:loc_x, loc_y - mh:loc_y, :] = mask[mask[:, :, 3] == 0][:, :, :3]
        if self.mask_cls == -1:
            self.label = label_new
        return self.img, self.label

    def loc_suit(self, mask, loc_x, loc_y, height, width, label_new):
        mh, mw = np.shape(mask)
        x1, y1, x2, y2 = [(loc_x - mw) / width, (loc_y - mh) / height, loc_x / width, loc_y / height]
        for l in label_new:
            if self.pub_area([x1, y1, x2, y2], l[1:]):
                return False
        return True

    def pub_area(self, rec1, rec2):
        rec2 = [rec2[0] - rec2[2] / 2, rec2[1] - rec2[3] / 2, rec2[0] + rec2[2] / 2, rec2[1] + rec2[3] / 2]
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        if left_line >= right_line or top_line >= bottom_line:
            return False
        return True
