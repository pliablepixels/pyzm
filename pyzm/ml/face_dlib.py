import numpy as np

import pyzm.ml.face_train_dlib as train

import dlib

import sys
import os
import cv2
import pickle
from sklearn import neighbors
import imutils
import math
import uuid
import time
import time as _time
import datetime
import logging
# Class to handle face recognition
import portalocker
import re

from pyzm.ml.face import Face

logger = logging.getLogger("pyzm")

_g_start = _time.perf_counter()
import face_recognition
g_diff_time = f"{(_time.perf_counter() - _g_start) * 1000:.2f} ms"

class FaceDlib(Face):
    def __init__(self, options={}):
        self.options = options
        global g_diff_time

        if dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() >=1 :
            self.processor = 'gpu'
        else:
            self.processor = 'cpu'
     
        logger.debug('perf: processor:{} Face Recognition library load time took: {} '.format(
                self.processor, g_diff_time))

        upsample_times = self.options.get('upsample_times',1)
        num_jitters= self.options.get('num_jitters',0)
        model=self.options.get('face_model','hog')

        logger.debug('Initializing face recognition with model:{} upsample:{}, jitters:{}'
            .format(model, upsample_times, num_jitters))

        self.disable_locks = options.get('disable_locks', 'no')

        self.upsample_times = upsample_times
        self.num_jitters = num_jitters
        if options.get('face_model'):
            self.face_model = options.get('face_model')
        else:
            self.face_model = model
       
        self.knn = None
        self.options = options
        self.is_locked = False

        self.lock_maximum=int(options.get(self.processor+'_max_processes') or 1)
        self.lock_timeout = int(options.get(self.processor+'_max_lock_wait') or 120)
        
        #self.lock_name='pyzm_'+self.processor+'_lock'
        self.lock_name='pyzm_uid{}_{}_lock'.format(os.getuid(),self.processor)
        if self.disable_locks == 'no':
            logger.debug('portalock: max:{}, name:{}, timeout:{}'.format(self.lock_maximum, self.lock_name, self.lock_timeout))
            self.lock = portalocker.BoundedSemaphore(maximum=self.lock_maximum, name=self.lock_name,timeout=self.lock_timeout)

        encoding_file_name = self.options.get('known_images_path') + '/faces.dat'
        try:
            if (os.path.isfile(self.options.get('known_images_path') +
                               '/faces.pickle')):
                # old version, we no longer want it. begone
                logger.debug('removing old faces.pickle, we have moved to clustering')
                os.remove(self.options.get('known_images_path') + '/faces.pickle')
        except Exception as e:
            logger.error('Error deleting old pickle file: {}'.format(e))

        # to increase performance, read encodings from  file
        if (os.path.isfile(encoding_file_name)):
            logger.debug('pre-trained faces found, using that. If you want to add new images, remove: {}'
                .format(encoding_file_name))

            #self.known_face_encodings = data["encodings"]
            #self.known_face_names = data["names"]
        else:
            # no encodings, we have to read and train
            logger.debug('trained file not found, reading from images and doing training...')
            logger.debug('If you are using a GPU and run out of memory, do the training using zm_train_faces.py. In this case, other models like yolo may already take up a lot of GPU memory')

            train.FaceTrain(options=self.options).train()
        try:
            with open(encoding_file_name, 'rb') as f:
                self.knn = pickle.load(f)
                f.close()
        except Exception as e:
            logger.error('Error loading KNN model: {}'.format(e))


    def get_options(self):
        return self.options
        
    def acquire_lock(self):
        if self.disable_locks=='yes':
            return
        if self.is_locked:
            logger.debug('{} portalock already acquired'.format(self.lock_name))
            return
        try:
            logger.debug('Waiting for {} portalock...'.format(self.lock_name))
            self.lock.acquire()
            logger.debug('Got {} lock...'.format(self.lock_name))
            self.is_locked = True

        except portalocker.AlreadyLocked:
            logger.error('Timeout waiting for {} portalock for {} seconds'.format(self.lock_name, self.lock_timeout))
            raise ValueError ('Timeout waiting for {} portalock for {} seconds'.format(self.lock_name, self.lock_timeout))


    def release_lock(self):
        if self.disable_locks=='yes':
            return
        if not self.is_locked:
            logger.debug('{} portalock already released'.format(self.lock_name))
            return
        self.lock.release()
        self.is_locked = False
        logger.debug('Released {} portalock'.format(self.lock_name))

    def get_classes(self):
        if self.knn:
            return self.knn.classes_
        else:
            return []

    def _rescale_rects(self, a):
        rects = []
        for (left, top, right, bottom) in a:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            rects.append([left, top, right, bottom])
        return rects

    

    def detect(self, image):
      
        Height, Width = image.shape[:2]
        logger.debug('|---------- Dlib Face recognition (input image: {}w*{}h) ----------|'.
            format(Width, Height))

        downscaled =  False
        upsize_xfactor = None
        upsize_yfactor = None
        max_size = self.options.get('max_size', Width)
        old_image = None

        logger.debug('Face options={}'.format(self.options))
        
        if Width > max_size:
            downscaled = True
            logger.debug('Scaling image down to max size: {}'.format(max_size))
            old_image = image.copy()
            image = imutils.resize(image,width=max_size)
            newHeight, newWidth = image.shape[:2]
            upsize_xfactor = Width/newWidth
            upsize_yfactor = Height/newHeight
        

        labels = []
        classes = []
        conf = []

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #rgb_image = image

        # Find all the faces and face encodings in the target image
        #prin (self.options)
        if self.options.get('auto_lock',True):
            self.acquire_lock()

        _t0 = _time.perf_counter()
        face_locations = face_recognition.face_locations(
            rgb_image,
            model=self.face_model,
            number_of_times_to_upsample=self.upsample_times)

        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
        logger.debug('perf: processor:{} Finding faces took {}'.format(self.processor, diff_time))

        _t0 = _time.perf_counter()
        face_encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=face_locations,
            num_jitters=self.num_jitters)

        if self.options.get('auto_lock',True):
            self.release_lock()

        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
        logger.debug('perf: processor:{} Computing face recognition distances took {}'.format(
                self.processor, diff_time))

        if not len(face_encodings):
            return [], [], [],[]

        # Use the KNN model to find the best matches for the test face
      
        logger.debug('Comparing to known faces...')

        _t0 = _time.perf_counter()
        if self.knn:
            closest_distances = self.knn.kneighbors(face_encodings, n_neighbors=1)
            logger.debug('Closest knn match indexes (lesser is better): {}'.format(closest_distances))
            are_matches = [
                closest_distances[0][i][0] <= float(self.options.get('face_recog_dist_threshold',0.6))
                for i in range(len(face_locations))
                
            ]
            prediction_labels = self.knn.predict(face_encodings)

        else:
            # There were no faces to compare
            # create a set of non matches for each face found
            are_matches = [False] * len(face_locations)
            prediction_labels = [''] * len(face_locations)
            logger.debug('No faces to match, so creating empty set')

        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
        logger.debug('perf: processor:{} Matching recognized faces to known faces took {}'.
            format(self.processor, diff_time))

        matched_face_names = []
        matched_face_rects = []


        if downscaled:
            logger.debug('Scaling image back up to {}'.format(Width))
            image = old_image
            new_face_locations = []
            for loc in face_locations:
                a,b,c,d=loc
                a = round(a * upsize_yfactor)
                b = round(b * upsize_xfactor)
                c = round(c * upsize_yfactor)
                d = round(d * upsize_xfactor)
                new_face_locations.append((a,b,c,d))
            face_locations = new_face_locations


        for pred, loc, rec in zip(prediction_labels,
                                  face_locations, are_matches):
            label = pred if rec else self.options.get('unknown_face_name', 'unknown')
            if not rec and self.options.get('save_unknown_faces') == 'yes':
                h, w, c = image.shape
                x1 = max(loc[3] - int(self.options.get('save_unknown_faces_leeway_pixels',0)),0)
                y1 = max(loc[0] - int(self.options.get('save_unknown_faces_leeway_pixels',0)),0)
                x2 = min(loc[1] + int(self.options.get('save_unknown_faces_leeway_pixels',0)), w)
                y2 = min(loc[2] + int(self.options.get('save_unknown_faces_leeway_pixels',0)),h)
                #print (image)
                crop_img = image[y1:y2, x1:x2]
                # crop_img = image
                timestr = time.strftime("%b%d-%Hh%Mm%Ss-")
                unf = self.options.get('unknown_images_path') + '/' + timestr + str(
                    uuid.uuid4()) + '.jpg'
                logger.info(
                    'Saving cropped unknown face at [{},{},{},{} - includes leeway of {}px] to {}'
                    .format(x1, y1, x2, y2,
                            self.options.get('save_unknown_faces_leeway_pixels'), unf))
                cv2.imwrite(unf, crop_img)

            
            matched_face_rects.append([loc[3], loc[0], loc[1], loc[2]])
            matched_face_names.append(label)
            #matched_face_names.append('face:{}'.format(label))
            conf.append(1)

        logger.debug('Face Dlib:Returning: {}, {}, {}'.format(matched_face_rects,matched_face_names, conf))
        return matched_face_rects, matched_face_names, conf, ['face_dlib']*len(matched_face_names)
