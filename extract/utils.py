import numpy as np
import cv2
from skimage import transform as trans

def distance2bbox(points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        print("--------------------")
        print(points.shape, distance.shape)
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def align_face(img, bbox=None, landmark=None, **kwargs): # bbox:x1,y1,x2,y2 landmakr (-1,5,2)
        M = None
        #x1, y1, x2, y2 = bbox
        #_size = max(x2-x1, y2-y1)
        #image_size = [int(_size), int(_size)]
        image_size = [112,112]

        if landmark is not None:
            assert len(image_size)==2
            # src = np.array([
            #   [30.2946, 51.6963],
            #   [65.5318, 51.5014],
            #   [48.0252, 71.7366],
            #   [33.5493, 92.3655],
            #   [62.7299, 92.2041] ], dtype=np.float32 )
            src = np.array([
            [30.2946*image_size[0]/112, 51.6963*image_size[1]/112],
            [65.5318*image_size[0]/112, 51.5014*image_size[1]/112],
            [48.0252*image_size[0]/112, 71.7366*image_size[1]/112],
            [33.5493*image_size[0]/112, 92.3655*image_size[1]/112],
            [62.7299*image_size[0]/112, 92.2041*image_size[1]/112] ], dtype=np.float32 )
        

            src[:,0] += 8.0*image_size[1]/112
            dst = landmark.astype(np.float32)
            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2,:]
           

        if M is None:
            if bbox is None: #use center crop
                det = np.zeros(4, dtype=np.int32)
                det[0] = int(img.shape[1]*0.0625)
                det[1] = int(img.shape[0]*0.0625)
                det[2] = img.shape[1] - det[0]
                det[3] = img.shape[0] - det[1]
            else:
                det = bbox
                margin = kwargs.get('margin', 44*image_size[0]/112)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0)
                bb[1] = np.maximum(det[1]-margin/2, 0)
                bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
                bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
                ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
                if len(image_size)>0:
                    ret = cv2.resize(ret, (image_size[1], image_size[0]))
            return ret 
        else: #do align using landmark
            assert len(image_size)==2
            warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
            return warped

def nms(dets, nms_thresh):
        thresh = nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
