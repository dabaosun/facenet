# coding=utf-8

import tornado.web
import tornado.ioloop
from tornado.web import Application
from tornado.ioloop import IOLoop
import facenet
import logging
import logging.config
import tensorflow as tf
import argparse
import configparser
import sys
import os
import face
import api

filepath = os.path.join(os.path.dirname(__file__), 'logging.conf')
logging.config.fileConfig(filepath)

logger = logging.getLogger("server")


class APIServer(object):
    def __init__(self, config):
        self.cfg = config

    def main(self):
        #self.model_checkpoint = self.cfg.get("model", "checkpoint")
        self.ip = self.cfg.get("server", "ip")
        self.port = self.cfg.getint("server", "port")

        app = make_app()
        app.init_detection(self.cfg)
        app.init_regconiztion(self.cfg)

        app.listen(self.port)
        logger.info("server started and binding on 3000 port")
        IOLoop.instance().start()


class ROIApplication(Application):
    def init_detection(self, config):
        self.detection = face.Detection()

    def init_regconiztion(self, config):
        self.recognition = face.Recognition(config)


def make_app():
    urls = [(r"/api/v1/face/detection", api.DetectFaceAPI),
            (r"/api/v1/face/signature", api.SignatureFaceAPI)]
    #settings={"detection":face.Detection()}
    return ROIApplication(urls)


def parse_arguments(argv):

    parser = argparse.ArgumentParser(description="Restful API server")
    parser.add_argument(
        '--config', type=str, help='config file', required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    cfg = configparser.ConfigParser()

    if cfg.read(os.path.join(os.path.dirname(__file__), 'roi.conf')):
        for each_section in cfg.sections():
            logger.info("[%s]" % (each_section))
            for (each_key, each_val) in cfg.items(each_section):
                logger.info("%s = %s" % (each_key, each_val))
        service = APIServer(cfg)
        service.main()
