# coding=utf-8
import tornado.web

from tornado.web import RequestHandler
import logging
logger = logging.getLogger("api")


class FaceHandler(RequestHandler):
    def data_received(self, chunk):
        pass

    def options(self):
        self.__set_response_header()

    def __set_response_header(self):
        self.set_header('content-type', 'application/json')
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Credentials", "true")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods',
                        'POST, GET, OPTIONS, HEAD')


class Face(FaceHandler):
    def get(self):
        self.set_status(405)

    def put(self):
        self.set_status(405)

    def delete(self):
        self.set_status(405)

    #@tornado.gen.coroutine
    def post(self, args=None):

        box = self.get_argument("box", None)
        body = self.request.body
        try:
            if box:
                logger.debug(box)

            if body:
                logger.debug(body)
            else:
                logger.error("not found body in request.")
                self.set_status(400, "not found body in request.")

        except Exception as err:
            logger.error(err)
            self.set_status(err.status_code)
            return
