import numpy as np

det = [[0, 0, 0, 0]]
res = []


def reflect(det, masks, mask_info):
    def pbp(det, range):
        x1, y1, x2, y2 = det
        x1_l, y1_l, x2_l, y2_l = range
        left  = max(x1, x1_l)
        right = min(x2, x2_l)
        top   = max(y1, y1_l)
        bot   = min(y2, y2_l)
        if left < right or top < bot:
            return 0
        else:
            all = (right-left)*(bot-top)
            inner =  (y2-y1)*(x2-x1)
            return inner / all
    max = 0
    more_possibiblty = -1
    for i, mask in enumerate(masks):
        pub = pbp(det, mask)
        if pub > max:
            max = pub
            more_possibiblty = i
    if more_possibiblty != -1:
        det = [max(det[0], mask[0]), max(det[1], mask[1]), min(det[2], mask[2]), min(det[3], mask[3])]
        para_det = [det[0] - mask[0], det[1]-mask[1], det[2]-mask[0], det[3]-mask[1]]
        abs_det = [para_det[0]+mask_info[more_possibiblty][0], para_det[1]+mask_info[more_possibiblty][1],
                   para_det[2] + mask_info[more_possibiblty][0], para_det[3] + mask_info[more_possibiblty][1]]
    else:
        abs_det = []
    return abs_det




# scale = [y_scale , x_scale]
def idea(fg, dets, inner_points, scale, masks, mask_info):
    for det in dets:
        sum_points = 0
        abs_det = reflect(det, masks, mask_info)
        x1, y1, x2, y2 = abs_det
        x1 *= scale[1]
        x2 *= scale[1]
        y1 *= scale[0]
        y2 *= scale[0]
        for i in range(int(y1) - 1, int(y2) + 1):
            for j in range(int(x1) - 1, int(x2) + 1):
                if fg[i, j] == 255:
                    sum_points += 1
        if sum_points >= inner_points:
            res.append(det)
    return np.array(res, dtype=np.uint8)
