import numpy as np


import sys
import os
import cv2

import math
import uuid
import time
import time as _time
import datetime
import logging
# Class to handle face recognition
import portalocker
import re
import imutils
from pyzm.ml.face import Face
from PIL import Image

logger = logging.getLogger("pyzm")

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

class FaceTpu(Face):
    def __init__(self, options={}):
        global g_diff_time
        self.options = options
      
        logger.debug('Initializing face detection')

        self.processor='tpu'
        self.lock_maximum=int(options.get(self.processor+'_max_processes') or 1)
        self.lock_name='pyzm_uid{}_{}_lock'.format(os.getuid(),self.processor)
        self.lock_timeout = int(options.get(self.processor+'_max_lock_wait') or 120)
        self.disable_locks = options.get('disable_locks', 'no')
        if self.disable_locks == 'no':
            logger.debug('portalock: max:{}, name:{}, timeout:{}'.format(self.lock_maximum, self.lock_name, self.lock_timeout))
            self.lock = portalocker.BoundedSemaphore(maximum=self.lock_maximum, name=self.lock_name,timeout=self.lock_timeout)
        self.is_locked = False
        self.model = None


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

    def load_model(self):
        name = self.options.get('name') or 'TPU'
        logger.debug('|--------- Loading "{}" model from disk -------------|'.format(name))

        _t0 = _time.perf_counter()
        self.model = make_interpreter(self.options.get('face_weights'))
        self.model.allocate_tensors()
        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
        logger.debug('perf: processor:{} TPU initialization (loading {} from disk) took: {}'
            .format(self.processor, self.options.get('face_weights'),diff_time))
        
    
    

    def detect(self, image):
        Height, Width = image.shape[:2]
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.options.get('auto_lock',True):
            self.acquire_lock()

        try:
            if not self.model:
                self.load_model()

            logger.debug('|---------- TPU (input image: {}w*{}h) ----------|'
                .format(Width, Height))
            _t0 = _time.perf_counter()
            _, scale = common.set_resized_input(
                self.model, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
            self.model.invoke()
            objs = detect.get_objects(self.model, float(self.options.get('face_min_confidence',0.1)), scale)

            diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"

            if self.options.get('auto_lock',True):
                self.release_lock()
        except:
            if self.options.get('auto_lock',True):
                self.release_lock()
            raise

        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
        logger.debug('perf: processor:{} Coral TPU detection took: {}'.format(self.processor, diff_time))
    
        bbox = []
        labels = []
        conf = []

        for obj in objs:
        # box = obj.bbox.flatten().astype("int")
            bbox.append([
                    int(round(obj.bbox.xmin)),
                    int(round(obj.bbox.ymin)),
                    int(round(obj.bbox.xmax)),
                    int(round(obj.bbox.ymax))
                ])
        
            labels.append(self.options.get('unknown_face_name', 'face'))
            conf.append(float(obj.score))
        logger.debug('Coral face is detection only. Skipping recognition phase')
        logger.debug('Coral face returning: {},{},{}'.format(bbox,labels,conf))
        return bbox, labels, conf, ['face_tpu'] * len(labels)

