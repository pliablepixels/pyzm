
import cv2
import argparse
import pickle
from sklearn import neighbors
import imutils
import math

import os
import time as _time
import logging

logger = logging.getLogger("pyzm")

import face_recognition

class FaceTrain:

    def __init__(self, options={}):
        self.options = options

    def train(self,size=None):
        _t0 = _time.perf_counter()
        known_images_path = self.options.get('known_images_path')
        train_model = self.options.get('face_train_model')
        knn_algo = self.options.get('face_recog_knn_algo', 'ball_tree') 
    
        upsample_times = int(self.options.get('face_upsample_times',1))
        num_jitters = int(self.options.get('face_num_jitters',0))

        encoding_file_name = known_images_path + '/faces.dat'
        try:
            if (os.path.isfile(known_images_path + '/faces.pickle')):
                # old version, we no longer want it. begone
                logger.debug('removing old faces.pickle, we have moved to clustering')
                os.remove(known_images_path + '/faces.pickle')
        except Exception as e:
            logger.error('Error deleting old pickle file: {}'.format(e))

        directory = known_images_path
        ext = ['.jpg', '.jpeg', '.png', '.gif']
        known_face_encodings = []
        known_face_names = []

        try:
            for entry in os.listdir(directory):
                if os.path.isdir(directory + '/' + entry):
                    # multiple images for this person,
                    # so we need to iterate that subdir
                    logger.debug('{} is a directory. Processing all images inside it'.
                        format(entry))
                    person_dir = os.listdir(directory + '/' + entry)
                    for person in person_dir:
                        if person.endswith(tuple(ext)):
                            logger.debug('loading face from  {}/{}'.format(
                                entry, person))

                            # imread seems to do a better job of color space conversion and orientation
                            known_face = cv2.imread('{}/{}/{}'.format(
                                directory, entry, person))
                            if known_face is None or known_face.size == 0:
                                logger.error('Error reading file, skipping')
                                continue
                            #known_face = face_recognition.load_image_file('{}/{}/{}'.format(directory,entry, person))
                            if not size:
                                size = int(self.options.get('resize',800))
                            logger.debug('resizing to {}'.format(size))
                            known_face = imutils.resize(known_face,width=size)

                            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                            known_face = cv2.cvtColor(known_face, cv2.COLOR_BGR2RGB)
                            
                            face_locations = face_recognition.face_locations(
                                known_face,
                                model=train_model,
                                number_of_times_to_upsample=upsample_times)
                            if len(face_locations) != 1:
                                logger.error(
                                    'File {} has {} faces, cannot use for training. We need exactly 1 face. If you think you have only 1 face try using "cnn" for training mode. Ignoring...'
                                .format(person, len(face_locations)))
                            else:
                                face_encodings = face_recognition.face_encodings(
                                    known_face,
                                    known_face_locations=face_locations,
                                    num_jitters=num_jitters)
                                known_face_encodings.append(face_encodings[0])
                                known_face_names.append(entry)

                elif entry.endswith(tuple(ext)):
                    # this was old style. Lets still support it. The image is a single file with no directory
                    logger.debug('loading face from  {}'.format(entry))
                    #known_face = cv2.imread('{}/{}/{}'.format(directory,entry, person))
                    known_face = cv2.imread('{}/{}'.format(directory, entry))

                    if not size:
                        size = int(self.options.get('resize',800))
                    logger.debug('resizing to {}'.format(size))
                    known_face = imutils.resize(known_face,width=size)
                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    known_face = known_face[:, :, ::-1]
                    face_locations = face_recognition.face_locations(
                        known_face,
                        model=train_model,
                        number_of_times_to_upsample=upsample_times)

                    if len(face_locations) != 1:
                        logger.error(
                                    'File {} has {} faces, cannot use for training. We need exactly 1 face. If you think you have only 1 face try using "cnn" for training mode. Ignoring...'
                                    .format(entry, len(face_locations)))
                    else:
                        face_encodings = face_recognition.face_encodings(
                            known_face,
                            known_face_locations=face_locations,
                            num_jitters=num_jitters)
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(os.path.splitext(entry)[0])

        except Exception as e:
            logger.error('Error initializing face recognition: {}'.format(e))
            raise ValueError(
                'Error opening known faces directory. Is the path correct?')

        # Now we've finished iterating all files/dirs
        # lets create the svm
        if not len(known_face_names):
            logger.error(
                'No known faces found to train, encoding file not created')
        else:
            n_neighbors = int(round(math.sqrt(len(known_face_names))))
            logger.debug('Using algo:{} n_neighbors to be: {}'.format(knn_algo, n_neighbors))
            knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                                algorithm=knn_algo,
                                                weights='distance')

            logger.debug('Training model ...')
            knn.fit(known_face_encodings, known_face_names)

            with open(encoding_file_name, "wb") as f:
                pickle.dump(knn, f)
            logger.debug('wrote encoding file: {}'.format(encoding_file_name))
        diff_time = f"{(_time.perf_counter() - _t0) * 1000:.2f} ms"
        logger.debug('perf: Face Recognition training took: {}'.format(diff_time))
