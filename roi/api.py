# coding=utf-8
import io
import logging
import face
import cv2
import PIL

import numpy as np
import urllib
import cv2

import urllib
import urllib.parse
import urllib.request
import json
import tornado.escape as escape

from tornado.web import RequestHandler
import face

logger = logging.getLogger("api")

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

    def post(self):
        self.options()

    def get(self):
        self.options()
        self.set_status(405)

    def put(self):
        self.options()
        self.set_status(405)

    def delete(self):
        self.options()
        self.set_status(405)

    def buildresponse(self, faces):
        faceslist = []
        for i, face in enumerate(faces):
            box = face.bounding_box.tolist()
            if not face.embedding is None:
                if (logger.getEffectiveLevel() == logging.DEBUG):
                    np.save("/tmp/face-{0}-{1}.npy", face.embedding)
                faceslist.append({
                    "point": {
                        "x": box[0],
                        "y": box[1],
                        "w": box[2],
                        "h": box[3]
                    },
                    "signature":
                    np.asarray(face.embedding).tolist()
                })
            else:
                faceslist.append({
                    "point": {
                        "x": box[0],
                        "y": box[1],
                        "w": box[2],
                        "h": box[3]
                    }
                })

        resp = {}
        resp['faces'] = faceslist
        resp['result'] = len(faces) > 0
        return json.dumps(resp, sort_keys=True)


class DetectFaceAPI(APIHandler):
    def post(self, args=None):
        super(DetectFaceAPI, self).post()
        body = self.request.body
        if not body:
            self.set_status(400, "not found body data in request.")
            return
        logger.debug(body)

        try:
            image = np.asarray(bytearray(body), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            faces = self.application.detection.find_faces(image)
            logger.debug("faces found : %d" % len(faces))
            self.write(self.buildresponse(faces))
        except (TypeError, AttributeError) as err:
            logger.error(err)
            self.set_status(400, str(err))
            return
        except Exception as err:
            logger.error(err)
            self.set_status(500, str(err))
            return


class SignatureFaceAPI(APIHandler):
    def post(self, args=None):
        super(SignatureFaceAPI, self).post()

        # boxes = self.get_argument("boxes", None)
        # if not boxes:
        #     self.set_status(400, "not found boxes json parameter in request.")
        #     return
        # logger.debug(boxes)

        body = self.request.body
        if not body:
            self.set_status(400, "not found body data in request.")
            return
        logger.debug(body)

        try:
            # boxes = json.loads(boxes)

            image = np.asarray(bytearray(body), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            faces = self.application.recognition.identify(image)
            self.write(self.buildresponse(faces))

        except (TypeError, AttributeError) as err:
            logger.error(err)
            self.set_status(400, str(err))
            return
        except Exception as err:
            logger.error(err)
            self.set_status(err.status_code)
            return
