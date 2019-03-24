# coding=utf-8
"""Face Detection and Recognition"""

import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import align.detect_face
import facenet

gpu_memory_fraction = 0.3
facenet_model_checkpoint = "/root/project/models/20180402-114759"  #os.path.dirname(__file__) + "/../model_checkpoints/20170512-110547"
debug = True


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()

    def identify(self, image):
        faces = self.detect.find_faces(image)

        for i, face in enumerate(faces):
            if debug:
                #cv2.imshow("Face: " + str(i), face.image)
                cv2.imwrite("/tmp/face.jpg", face.image)
            face.embedding = self.encoder.generate_embedding(face)

        return faces


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name(
            "input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name(
            "phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {
            images_placeholder: [prewhiten_face],
            phase_train_placeholder: False
        }
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(
                config=tf.ConfigProto(
                    gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = align.detect_face.detect_face(
            image, self.minsize, self.pnet, self.rnet, self.onet,
            self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(
                bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(
                bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(
                bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(
                bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.
                            bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(
                cropped, (self.face_crop_size, self.face_crop_size),
                interp='bilinear')

            faces.append(face)

        return faces
