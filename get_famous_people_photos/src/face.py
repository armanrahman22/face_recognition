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
from memory_profiler import profile
from scipy import misc

facenet_model_checkpoint = "../../common/models/20180402-114759.pb"
debug = False


class Face:
    """Class representing a single face
    """
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.matches = []
        self.url = None


class Match:
    """Class representing a match between two faces
    """
    def __init__(self):
        self.face_1 = Face()
        self.face_2 = Face()
        self.score = float("inf")
        self.is_match = False


class Identifier:
    """Class to detect, encode, and match faces

    Arguments:
        threshold {Float} -- Distance threshold to determine matches
    """
    def __init__(self, threshold=1.10):
        self.detector = Detector()
        self.encoder = Encoder()
        self.threshold = threshold
    
    def download_image(self, url):
        """Downloads an image from the url as a cv2 image
        
        Arguments:
            url {str} -- url of image
        
        Returns:
            cv2 image -- image array
        """

        req = urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)
        return image
    
    def detect_encode(self, image, face_limit=5):
        """Detects faces in an image and encodes them
        
        Arguments:
            image {cv2 image (np array)} -- image to find faces and encode
            face_limit {int} -- Maximum # of faces allowed in image. 
                                If over limit returns empty list 

        Returns:
            Face[] -- list of Face objects with embeddings attached
        """

        faces = self.detector.find_faces(image, face_limit)
        for face in faces:
            face.embedding = self.encoder.generate_embedding(face)
        return faces
    
    def detect_encode_all(self, images, urls=None, save_memory=False):
        """For a list of images finds and encodes all faces
        
        Arguments:
            images {List or iterable of cv2 images} -- images to encode
        
        Keyword Arguments:
            urls {str[]} -- Optional list of urls to attach to Face objects. 
                            Should be same length as images if used. (default: {None})
            save_memory {bool} -- Saves memory by deleting image array from Face objects.
                                  Should only be used if with r(default: {False})
        
        Returns:
            [type] -- [description]
        """

        all_faces = self.detector.bulk_find_face(images, urls)
        all_embeddings = self.encoder.get_all_embeddings(all_faces, save_memory)
        return all_embeddings
    
    def compare_embedding(self, embedding_1, embedding_2):
        distance = facenet.distance(embedding_1.reshape(1,-1), embedding_2.reshape(1,-1), distance_metric=0)[0]
        is_match = False
        if distance < self.threshold:
            is_match = True
        return is_match, distance

    def compare_images(self, image_1, image_2):
        match = Match()
        image_1_faces = self.detect_encode(image_1)
        image_2_faces = self.detect_encode(image_2)
        if image_1_faces and image_2_faces:
            for face_1 in image_1_faces:
                for face_2 in image_2_faces:
                    distance = facenet.distance(face_1.embedding.reshape(1,-1), face_2.embedding.reshape(1,-1), distance_metric=0)[0]
                    if distance < match.score:
                        match.score = distance
                        match.face_1 = face_1
                        match.face_2 = face_2
            if distance < self.threshold:
                match.is_match = True
        return match
    
    def find_all_matches(self, image_directory):
        all_images = glob(image_directory + '/*')
        all_matches = []
        all_faces = self.detect_encode_all(all_images)
        # Really inefficient way to check all combinations
        for face_1, face_2 in itertools.combinations(all_faces, 2):
            is_match, score = self.compare_embedding(face_1.embedding, face_2.embedding)
            if is_match:
                match = Match()
                match.face_1 = face_1
                match.face_2 = face_2
                match.is_match = True 
                match.score = score
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
    
    def get_all_embeddings(self, all_faces, save_memory=False):
        all_images = [facenet.prewhiten(face.image) for face in all_faces]

        # Run forward pass to calculate embeddings
        feed_dict = {self.images_placeholder: all_images, self.phase_train_placeholder: False}
        embed_array = self.sess.run(self.embeddings, feed_dict=feed_dict)
        
        for index, face in enumerate(all_faces):
            if save_memory:
                face.image = None
            face.embedding = embed_array[index]
        return all_faces


class Detector:
    # face detection parameters
    def __init__(self, face_crop_size=160, face_crop_margin=32, gpu_memory_fraction=0.4, detect_multiple_faces=True):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn(gpu_memory_fraction)
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.detect_multiple_faces = detect_multiple_faces

    def _setup_mtcnn(self, gpu_memory_fraction):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, None)
    
    def get_face_from_bb(self, bounding_boxes, img, min_size=50):
        nrof_faces = bounding_boxes.shape[0]
        faces = []
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                if self.detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    img_center = img_size / 2
                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-self.face_crop_margin/2, 0)
                bb[1] = np.maximum(det[1]-self.face_crop_margin/2, 0)
                bb[2] = np.minimum(det[2]+self.face_crop_margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+self.face_crop_margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                if cropped.shape[0] > min_size and cropped.shape[1] > min_size:
                    face = Face()
                    face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
                    faces.append(face)
        return faces

    def bulk_find_face(self, images, urls=None, face_limit=5):
        all_faces = []
        for index, image in enumerate(images):
            bb, _ = detect_face.detect_face(image, self.minsize,
                                        self.pnet, self.rnet, self.onet,
                                        self.threshold, self.factor)
            if urls and index < len(urls):
                faces = self.get_face_from_bb(bb, image)
                if len(faces) < face_limit:
                    for face in faces:
                        face.url = urls[index]
                        all_faces.append(face)
            else:
                faces = self.get_face_from_bb(bb, image)
                all_faces.append(faces)
        return all_faces 
    
    def find_faces(self, image, face_limit=5):
        if type(image) == str:
            image = cv2.imread(image)
        faces = []
        bounding_boxes, _ = detect_face.detect_face(image, self.minsize,
                                                           self.pnet, self.rnet, self.onet,
                                                           self.threshold, self.factor)
        faces = self.get_face_from_bb(bounding_boxes, image)
        if len(faces) > face_limit:
            return []
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
