# coding=utf-8

import argparse
import sys

from scipy import misc

import face


def add_overlays(faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
   

def main(args):

    img = misc.imread(
        "/root/project/roinet/data/images/Anthony_Hopkins_0001.jpg",
        mode='RGB')

    face_recognition = face.Recognition()

    if args.debug:
        print("Debug enabled")
        face.debug = True
    img1 = cv2.imread(
        "/root/project/roinet/data/images/Anthony_Hopkins_0001.jpg")
    cv2.imshow("Face: ", img1)

    faces = face_recognition.identify(img)

    add_overlays(faces)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--debug', action='store_true', help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
