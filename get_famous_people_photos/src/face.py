# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import itertools
import os
import pickle
from glob import glob
from urllib.request import urlopen

import cv2
import numpy as np
import tensorflow as tf
from facenet_sandberg import facenet, validate_on_lfw
from facenet_sandberg.align import align_dataset_mtcnn, detect_face
from scipy import misc

facenet_model_checkpoint = "../../common/models/20180402-114759.pb"
debug = False


class Face:
    """
    Class representing a single face
    """
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.matches = []


class Match:
    """
    Class representing a single face
    """
    def __init__(self):
        self.face_1 = Face()
        self.face_2 = Face()
        self.score = float("inf")
        self.is_match = False


class Identifier:
    def __init__(self, threshold=1.10):
        self.detector = Detector()
        self.encoder = Encoder()
        self.threshold = threshold
    
    def detect_encode(self, image):
        faces = self.detector.find_faces(image)
        for face in faces:
            face.embedding = self.encoder.generate_embedding(face)
        return faces
    
    def compare_faces(self, image_1, image_2, is_from_url):
        match = Match()
        if is_from_url:
            req = urlopen(image_1)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            image_1 = cv2.imdecode(arr, -1)
            req = urlopen(image_2)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            image_2 = cv2.imdecode(arr, -1)
        image_1_faces = self.detect_encode(image_1)
        image_2_faces = self.detect_encode(image_2)
        if image_1_faces and image_2_faces:
            for face_1 in image_1_faces:
                for face_2 in image_2_faces:
                    distance = np.sqrt(np.sum(np.square(np.subtract(face_1.embedding, face_2.embedding))))
                    if distance < match.score:
                        match.score = distance
                        match.face_1 = face_1
                        match.face_2 = face_2
            if distance < self.threshold:
                match.is_match = True
        return match
    
    def find_all_matches(self, image_directory):
        all_images = glob(image_directory + '/*')
        all_faces = []
        all_matches = []
        for image in all_images:
            all_faces += self.detect_encode(image)
        for face_1, face_2 in itertools.combinations(all_faces, 2):
            match = Match()
            match.face_1 = face_1
            match.face_2 = face_2
            match.score = np.sqrt(np.sum(np.square(np.subtract(face_1.embedding, face_2.embedding))))
            if match.score < self.threshold:
                match.is_match = True 
                all_matches.append(match)
                face_1.matches.append(match)
                face_2.matches.append(match)
        return all_faces, all_matches


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)
        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    def generate_embedding(self, face):
        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {self.images_placeholder: [prewhiten_face], self.phase_train_placeholder: False}
        return self.sess.run(self.embeddings, feed_dict=feed_dict)[0]
    


class Detector:
    # face detection parameters
    def __init__(self, face_crop_size=160, face_crop_margin=32, gpu_memory_fraction=0.4):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn(gpu_memory_fraction)
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor

    def _setup_mtcnn(self, gpu_memory_fraction):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        if type(image) == str:
            image = cv2.imread(image)
        faces = []
        bounding_boxes, _ = detect_face.detect_face(image, self.minsize,
                                                           self.pnet, self.rnet, self.onet,
                                                           self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            faces.append(face)
        if debug:
            for i, face in enumerate(faces):
                cv2.imshow("Face: " + str(i), face.image)
        return faces


def align_dataset(input_dir, output_dir, image_size, margin, random_order, gpu_memory_fraction, detect_multiple_faces):
    args = [input_dir, output_dir, '--image_size', image_size, '--margin', margin]
    if random_order:
        args.append('--random_order')
    args.append('--gpu_memory_fraction')
    args.append(gpu_memory_fraction)
    if detect_multiple_faces:
        args.append('--detect_multiple_faces')
    align_dataset_mtcnn.main(align_dataset_mtcnn.parse_arguments(args))

# def test_dataset(input_dir, output_dir, image_size, margin, random_order, gpu_memory_fraction, detect_multiple_faces):
#     args = [input_dir, output_dir, '--image_size', image_size, '--margin', margin]
#     if random_order:
#         args.append('--random_order')
#     args.append('--gpu_memory_fraction')
#     args.append(gpu_memory_fraction)
#     if detect_multiple_faces:
#         args.append('--detect_multiple_faces')
#     align_dataset_mtcnn.main(align_dataset_mtcnn.parse_arguments(args))
