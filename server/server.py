# coding: utf-8
import os

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
from tornado.escape import json_decode, json_encode
from tornado.concurrent import Future

from object_detection.rfcn_detection import rfcn_model_instance
from classifier import classifier as clf

define("port", default=8000, help="run on the given port", type=int)

class ClothHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.cloth_service = ClothService()

    @tornado.gen.coroutine
    def post(self):
        # [Request] (multipart/form-data)
        # {
        #     "name": "img",
        #     "file": "xxx.jpg"
        # }

        file_metas = self.request.files.get("img")

        # 上传图片，返回图片路径
        image_path = yield self.cloth_service.upload_image(file_metas)
        print("---- 文件上传完成，正在边框检测 ----")
        
        if image_path:
            bboxs = yield self.cloth_service.detection_model_run(image_path)
            print("---- bounding box 检测完成，正在分类搜索 ----")
            res = yield self.cloth_service.classifier_model_run(image_path, bboxs)
            print("---- 分类检测完成 ----")
        else:
            res = dict(
                rtn = 500,
                msg = "文件上传出错",
                data = {}
            )

        self.set_status(res.get("rtn"))
        self.set_header("Content-Type", "application/json")
        self.write(json_encode(res))
        self.finish()

# Service
class ClothService(object):
    def upload_image(self, file_metas):
        res_future = Future()

        file_path = None
        if (file_metas):
            for meta in file_metas:
                upload_path = os.path.join(os.path.dirname(__file__), "realimg")
                
                filename = meta['filename']
                file_path = os.path.join(upload_path, filename)

                with open(file_path, 'wb') as f:
                    f.write(meta['body'])
        
        res_future.set_result(file_path)
        print(file_path)
        
        return res_future
    
    
    def detection_model_run(self, image_path):
        bboxs_future = Future()
        try:
            bboxs = rfcn_model_instance.detection_image(image_path)
            bboxs = list(map(lambda x: dict(
                label = x.get("class"),
                score = x.get("score"),
                bbox = dict(
                    x_min = x["bbox"][0],
                    y_min = x["bbox"][1],
                    x_max = x["bbox"][2],
                    y_max = x["bbox"][3]
                )
            ), bboxs))

    
        except Exception as e:
            bboxs = []
        
        bboxs_future.set_result(bboxs) 
        
        return bboxs_future

    def classifier_model_run(self, image_path, bboxs):
        res_future = Future()
        res = dict(
            rtn = 200,
            msg = "",
            data = {}
        )

        res_images = []
        for bbox_data in bboxs:
            bbox = bbox_data.get("bbox")
            images = (clf.similar_cloth(image_path, \
                           bbox.get("y_min", 0), bbox.get("x_min", 0),\
                           bbox.get("y_max", 0), bbox.get("x_max", 0)))
            
            images = list(map(lambda x: "/" + str(x), images))
            
            res_images += images 

        print(res_images)

        if (len(res_images) > 0):
            res["msg"] = "get images path"
            res["data"] = dict(
                images = res_images
            )
        else:
            res["rtn"] = 404
            res["msg"] = "未找到任何相似图片"
        
        res_future.set_result(res)

        return res_future

static_path = os.path.join(os.path.dirname(__file__), "static")

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[
        (r"/api/cloth_search", ClothHandler),
        (r"/img/(.*)", tornado.web.StaticFileHandler, {"path": "./img/"})
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
