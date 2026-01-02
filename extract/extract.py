# from scrfd_engine import SCRFDTRT
from scr_onnx import SCRFD
from arcface_engine import IRES
from utils import *
import os
import base64

class SaveFeature:
    def __init__(self):

        # self.scrfd = SCRFDTRT()
        self.iresnet = IRES("/home/mq/disk2T/quangnv/models/arcface/arcface_r100.onnx")
        self.scrfd = SCRFD(model_file='/home/mq/disk2T/quangnv/models/scrfd640/scrfd_2.5g_bnkps_dynamic.onnx')
        self.scrfd.prepare(-1)

    def save(self, name, image, dict_save):
        # bboxes, kpss = self.scrfd.predict(image)
        bboxes, kpss = self.scrfd.detect(image, 0.5, input_size = (640, 640))
        if len(bboxes) == 1:
            x1, y1, x2, y2 = bboxes[0][:4]
            lm = kpss[0]
            face_aligned = align_face(image.copy(), [x1, y1, x2, y2], lm)
            feature = self.iresnet.predict(face_aligned.copy())

            # Encode face as base64 avatar (JPEG for smaller size)
            _, buffer = cv2.imencode('.jpg', face_aligned, [cv2.IMWRITE_JPEG_QUALITY, 85])
            avatar_base64 = base64.b64encode(buffer).decode('utf-8')

            face_aligned[65:, 65:, :] = 0
            feature_mask = self.iresnet.predict(face_aligned)
            dict_save[name] = {
                "feature": feature.tolist(),
                "feature_mask": feature_mask.tolist(),
                "avatar": avatar_base64
            }

import json
if __name__ == "__main__" :
    save_fea = SaveFeature()
    dict_save = {}
    for img_path in os.listdir("pics/"):
        name = img_path.split('.')[0]
        print("-------->", name)
        img = cv2.imread("pics/" + img_path)
        save_fea.save(name, img, dict_save)
    
    json_object = json.dumps(dict_save, indent=4, ensure_ascii=False)
    with open("features_arcface.json", "w",  encoding='utf8') as outfile:
        outfile.write(json_object)