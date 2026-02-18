import logging

logger = logging.getLogger("pyzm")


class Face:
    def __init__(self, options={}):

        self.model = None
        self.options = options
        if self.options.get('face_detection_framework') == 'dlib':
            import pyzm.ml.face_dlib as face_dlib
            self.model = face_dlib.FaceDlib(self.options)
        elif self.options.get('face_detection_framework') == 'tpu':
            import pyzm.ml.face_tpu as face_tpu
            self.model = face_tpu.FaceTpu(self.options)
        else:
            raise ValueError ('{} face detection framework is unknown'.format(self.options.get('face_detection_framework')))
   
    def detect(self, image):
        return self.model.detect(image)
    
    def get_options(self):
        return self.model.get_options()

    @property
    def lock_name(self):
        return self.model.lock_name

    def acquire_lock(self):
        return self.model.acquire_lock()

    def release_lock(self):
        return self.model.release_lock()

    def load_model(self):
        return self.model.load_model()
        