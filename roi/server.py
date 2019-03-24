# coding=utf-8
import api
import tornado.web
import tornado.ioloop
from tornado.web import Application
from tornado.ioloop import IOLoop

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("server")


def make_app():
    urls = [(r"/api/v1/face/detection", api.DetectFaceAPI),
            (r"/api/v1/face/recognition", api.RecognizeFaceAPI)]
    return Application(urls, debug=True)


if __name__ == '__main__':
    app = make_app()
    app.listen(3000)
    logger.info("server started and binding on 3000 port")
    IOLoop.instance().start()
