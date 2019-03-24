# coding=utf-8
import tornado.web

from tornado.web import RequestHandler
import logging
import face
import cv2
import numpy
import PIL
import io

import numpy as np
import urllib
import cv2

import urllib
import urllib.parse
import urllib.request
import json
import tornado.escape as escape
logger = logging.getLogger("api")

face_recognition = face.Recognition()
face_detectiion = face.Detection()


class APIHandler(RequestHandler):
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

    def get(self):
        self.set_status(405)

    def put(self):
        self.set_status(405)

    def delete(self):
        self.set_status(405)


class DetectFaceAPI(APIHandler):
    def post(self, args=None):
        body = self.request.body
        try:

            if body:
                logger.debug(body)
                image = np.asarray(bytearray(body), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                faces = face_detectiion.find_faces(image)
                logger.debug("faces found : %d" % len(faces))

                t = {}
                p = []
                for face in faces:
                    f = {}
                    box = face.bounding_box.tolist()
                    f['x'] = box[0]
                    f['y'] = box[1]
                    f['w'] = box[2]
                    f['h'] = box[3]
                    p.append(f)
                t['faces'] = p
                t['result'] = len(faces) > 0

                self.write(json.dumps(t, sort_keys=False))
            else:
                logger.error("not found body in request.")
                self.set_status(400, "not found body data in request.")
        except (TypeError, AttributeError) as err:
            logger.error(err)
            self.set_status(400, "body type invalid.")
            return
        except Exception as err:
            logger.error(err)
            self.set_status(500, str(err))
            return


class RecognizeFaceAPI(APIHandler):
    def post(self, args=None):

        box = self.get_argument("box", None)
        body = self.request.body
        try:

            if body:
                logger.debug(body)
                image = np.asarray(bytearray(body), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                if box:
                    logger.debug(box)
                    faces = face_recognition.identify(image)
                    add_overlays(faces)

                else:

                    faces = face_recognition.identify(image)
                    add_overlays(faces)
            else:
                logger.error("not found body in request.")
                self.set_status(400, "not found body in request.")

        except Exception as err:
            logger.error(err)
            self.set_status(err.status_code)
            return
