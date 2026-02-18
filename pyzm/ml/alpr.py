import logging
import numpy as np
import cv2
import requests
import os
import imutils
import json
import base64
import subprocess
import uuid

logger = logging.getLogger("pyzm")


class AlprBase:
    def __init__(self,options={}, tempdir='/tmp'):

        if not options.get('alpr_key'):
            logger.debug('No API key specified, hopefully you do not need one')
        self.apikey = options.get('alpr_key')
        self.tempdir = tempdir
        self.url = options.get('alpr_url')
        self.options = options
        self.disable_locks = options.get('disable_locks', 'no')
        name = self.options.get('name') or 'ALPR'
        logger.debug('Initializing ALPR:{} with options:{}'.format(name, self.options))

    def get_options(self):
        return self.options

    @property
    def lock_name(self):
        return None

    def acquire_lock(self):
        pass

    def release_lock(self):
        pass

    def load_model(self):
        pass

    def setkey(self, key=None):
        self.apikey = key
        logger.debug('Key changed')

    def stats(self):
        logger.debug('stats not implemented in base class')

    def detect(self, image=None):
        logger.debug('detect not implemented in base class')

    def prepare(self, object):
        if not isinstance(object, str):
            logger.debug('Supplied object is not a file, assuming blob and creating file')
            if self.options.get('max_size'):
                logger.debug('resizing image blob to {}'.format(self.options.get('max_size')))
                obj_new = imutils.resize(object, width=min(int(self.options.get('max_size')),
                                               object.shape[1]))
                object = obj_new
            # use png so there is no loss
            self.filename = self.tempdir+'/'+str(uuid.uuid4())+'-alpr.png'
            cv2.imwrite(self.filename, object)
            self.remove_temp = True
        else:
            # If it is a file and zm_detect sent it, it would already be resized
            # If it is a file and zm_detect did not send it, no need to adjust scales
            # as there won't be a yolo/alpr size mismatch
            logger.debug('supplied object is a file {}'.format(object))
            self.filename = object
            self.remove_temp = False

    def getscale(self):
        if self.options.get('resize') and self.options.get('resize') != 'no':
            img = cv2.imread(self.filename)
            img_new = imutils.resize(img,
                                     width=min(int(self.options.get('resize')),
                                               img.shape[1]))
            oldh, oldw, _ = img.shape
            newh, neww, _ = img_new.shape
            rescale = True
            xfactor = neww / oldw
            yfactor = newh / oldh
            img = None
            img_new = None
            logger.debug(
                'ALPR will use {}x{} but Object uses {}x{} so ALPR boxes will be scaled {}x and {}y'
                .format(oldw, oldh, neww, newh, xfactor, yfactor))
        else:
            xfactor = 1
            yfactor = 1
        return (xfactor, yfactor)


class Alpr(AlprBase):
    def __init__(self, options={}, tempdir='/tmp'):
        """Wrapper class for all ALPR objects

        Args:
            options (dict, optional): Config options. Defaults to {}.
            tempdir (str, optional): Path to store image for analysis. Defaults to '/tmp'.
        """
        AlprBase.__init__(self, options=options, tempdir=tempdir)
        self.alpr_obj = None

        if self.options.get('alpr_service') == 'plate_recognizer':
            self.alpr_obj = PlateRecognizer(options=self.options)
        elif self.options.get('alpr_service') == 'open_alpr':
            self.alpr_obj = OpenAlpr(options=self.options)
        elif self.options.get('alpr_service') == 'open_alpr_cmdline':
            self.alpr_obj = OpenAlprCmdLine(options=self.options)
        else:
            raise ValueError('ALPR service "{}" not known'.format(self.options.get('alpr_service')))

    def detect(self, image=None):
        return self.alpr_obj.detect(image)

    def stats(self):
        return self.alpr_obj.stats()



class PlateRecognizer(AlprBase):

    def __init__(self, options={}, tempdir='/tmp'):
        """Wrapper class for platerecognizer.com

        Args:
            options (dict, optional): Config options. Defaults to {}.
            tempdir (str, optional): Path to store image for analysis. Defaults to '/tmp'.
        """
        AlprBase.__init__(self, options=options,  tempdir=tempdir)

        url=self.options.get('alpr_url')
        apikey=self.options.get('alpr_key')

        if not url:
            self.url = 'https://api.platerecognizer.com/v1'

        logger.debug('PlateRecognizer ALPR initialized with url: {}'.format(self.url))

    def stats(self):
        """Returns API statistics

        Returns:
            HTTP Response: HTTP response of statistics API
        """
        if self.options.get('alpr_api_type') != 'cloud':
            logger.debug('local SDK does not provide stats')
            return {}
        try:
            if self.apikey:
                headers={'Authorization': 'Token ' + self.apikey}
            else:
                headers={}
            response = requests.get(self.url + '/statistics/', headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            response = {'error': str(e)}
        else:
            response = response.json()
        return response

    def detect(self, image=None):
        """Detects license plate using platerecognizer

        Args:
            object (image): image buffer

        Returns:
            boxes, labels, confidences: 3 objects, containing bounding boxes, labels and confidences
        """
        object = image
        bbox = []
        labels = []
        confs = []
        self.prepare(object)
        if self.options.get('platerec_stats') == 'yes':
            logger.debug('Plate Recognizer API usage stats: {}'.format(
                json.dumps(self.stats())))
        with open(self.filename, 'rb') as fp:
            try:
                platerec_url = self.url
                if self.options.get('alpr_api_type') == 'cloud':
                    platerec_url += '/plate-reader'

                platerec_payload = {}
                platerec_config = None
                if self.options.get('platerec_regions'):
                    platerec_payload['regions'] = self.options.get('platerec_regions')
                if self.options.get('platerec_payload'):
                    logger.debug('Found platerec_payload, overriding payload with values provided inside it')
                    platerec_payload = self.options.get('platerec_payload')
                if self.options.get('platerec_config'):
                    logger.debug('Found platerec_config, using it')
                    platerec_payload['config']= json.dumps(self.options.get('platerec_config'))
                response = requests.post(
                   platerec_url,
                   timeout=15,
                    #self.url ,
                    files=dict(upload=fp),
                    data=platerec_payload,
                    headers={'Authorization': 'Token ' + self.apikey})
                fp.close()
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                c = response.content
                response = {
                    'error':
                    'Plate recognizer rejected the upload with: {}.'.format(e),
                    'results': []
                }
                logger.error(
                    'Plate recognizer rejected the upload with {} and body:{}'.format(e,c)
                )
            else:
                response = response.json()
                logger.debug('ALPR JSON: {}'.format(response))

        #(xfactor, yfactor) = self.getscale()

        if self.remove_temp:
            os.remove(self.filename)

        if response.get('results'):
            for plates in response.get('results'):
                label = plates['plate']
                dscore = plates['dscore']
                score = plates['score']
                if dscore >= self.options.get('platerec_min_dscore') and score >= self.options.get(
                        'platerec_min_score'):
                    x1 = round(int(plates['box']['xmin']))
                    y1 = round(int(plates['box']['ymin']))
                    x2 = round(int(plates['box']['xmax']))
                    y2 = round(int(plates['box']['ymax']))
                    labels.append('alpr:{}'.format(label))
                    bbox.append([x1, y1, x2, y2])
                    confs.append(plates['score'])
                else:
                    logger.debug(
                        'ALPR: discarding plate:{} because its dscore:{}/score:{} are not in range of configured dscore:{} score:{}'
                        .format(label, dscore, score, self.options.get('platerec_min_dscore'),
                                self.options.get('platerec_min_score')))

        logger.debug('Exiting ALPR with labels:{}'.format(labels))
        return (bbox, labels, confs, ['platerec'] * len(labels))


class OpenAlpr(AlprBase):
    def __init__(self, options={}, tempdir='/tmp'):
        """Wrapper class for Open ALPR service

        Args:
            options (dict, optional): Various ALPR options. Defaults to {}.
            tempdir (str, optional): Temporary path to analyze image. Defaults to '/tmp'.
        """
        AlprBase.__init__(self, options=options, tempdir=tempdir)
        if not url:
            self.url = 'https://api.openalpr.com/v2/recognize'

        logger.debug('Open ALPR initialized with url: {}'.format(self.url))


    def detect(self, image=None):
        """Detection using OpenALPR

        Args:
            object (image): image buffer

        Returns:
            boxes, labels, confidences: 3 objects, containing bounding boxes, labels and confidences
        """
        object = image
        bbox = []
        labels = []
        confs = []

        self.prepare(object)

        with open(self.filename, 'rb') as fp:
            try:

                params = ''
                if self.options.get('openalpr_country'):
                    params = params + '&country=' + self.options.get('oenalpr_country')
                if self.options.get('openalpr_state'):
                    params = params + '&state=' + self.options.get('openalpr_state')
                if self.options.get('openalpr_recognize_vehicle'):
                    params = params + '&recognize_vehicle=' + \
                        str(self.options.get('openalpr_recognize_vehicle'))

                rurl = '{}?secret_key={}{}'.format(self.url, self.apikey,
                                                   params)
                logger.debug('Trying OpenALPR with url:' + rurl)
                response = requests.post(rurl, files={'image': fp})
                fp.close()
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                response = {
                    'error':
                    'Open ALPR rejected the upload with {}'.format(e),
                    'results': []
                }
                logger.debug('Open APR rejected the upload with {}'.format(e))
            else:
                response = response.json()
                logger.debug('OpenALPR JSON: {}'.format(response))

        #(xfactor, yfactor) = self.getscale()

        rescale = False

        if self.remove_temp:
            os.remove(filename)

        if response.get('results'):
            for plates in response.get('results'):
                label = plates['plate']
                conf = float(plates['confidence']) / 100
                if conf < float(options.get('openalpr_min_confidence')):
                    logger.debug(
                        'OpenALPR: discarding plate: {} because detected confidence {} is less than configured min confidence: {}'
                        .format(label, conf, self.options.get('openalpr_min_confidence')))
                    continue

                if plates.get(
                        'vehicle'):  # won't exist if recognize_vehicle is off
                    veh = plates.get('vehicle')
                    for attribute in ['color', 'make', 'make_model', 'year']:
                        if veh[attribute]:
                            label = label + ',' + veh[attribute][0]['name']

                x1 = round(int(plates['coordinates'][0]['x']))
                y1 = round(int(plates['coordinates'][0]['y']))
                x2 = round(int(plates['coordinates'][2]['x']))
                y2 = round(int(plates['coordinates'][2]['y']))
                labels.append('alpr:{}'.format(label))
                bbox.append([x1, y1, x2, y2])
                confs.append(conf)

        return (bbox, labels, confs,['openalpr'] * len(labels))

class OpenAlprCmdLine(AlprBase):
    def __init__(self, options={}, tempdir='/tmp'):
        """Wrapper class for OpenALPR command line utility

        Args:
            cmd (string, optional): The cli command. Defaults to None.
            options (dict, optional): Various ALPR options. Defaults to {}.
            tempdir (str, optional): Temporary path to analyze image. Defaults to '/tmp'.
        """
        AlprBase.__init__(self, options=options, tempdir=tempdir)

        cmd=self.options.get('openalpr_cmdline_binary')


        self.cmd = cmd + ' ' + self.options.get('openalpr_cmdline_params')
        if self.cmd.lower().find('-j') == -1:
            logger.debug('Adding -j to OpenALPR for json output')
            self.cmd = self.cmd + ' -j'


    def detect(self, image = None):
        """Detection using OpenALPR command line

        Args:
            object (image): image buffer

         Returns:
            boxes, labels, confidences: 3 objects, containing bounding boxes, labels and confidences
        """
        object = image
        bbox = []
        labels = []
        confs = []

        self.prepare(object)
        self.cmd = self.cmd + ' ' + self.filename
        logger.debug('OpenALPR CmdLine Executing: {}'.format(self.cmd))
        response = subprocess.check_output(self.cmd, shell=True)
        logger.debug('OpenALPR CmdLine Response: {}'.format(response))
        try:
            response = json.loads(response)
        except ValueError as e:
            logger.error('Error parsing JSON from command line: {}'.format(e))
            response = {}

        #(xfactor, yfactor) = self.getscale()

        rescale = False

        if self.remove_temp:
            os.remove(self.filename)

        if response.get('results'):
            for plates in response.get('results'):
                label = plates['plate']
                conf = float(plates['confidence']) / 100
                if conf < float(self.options.get('openalpr_cmdline_min_confidence')):
                    logger.debug(
                        'OpenALPR cmd line: discarding plate: {} because detected confidence {} is less than configured min confidence: {}'
                        .format(label, conf, self.options.get('openalpr_cmdline_min_confidence')))
                    continue

                x1 = round(int(plates['coordinates'][0]['x']))
                y1 = round(int(plates['coordinates'][0]['y']))
                x2 = round(int(plates['coordinates'][2]['x']))
                y2 = round(int(plates['coordinates'][2]['y']))
                labels.append('alpr:{}'.format(label))
                bbox.append([x1, y1, x2, y2])
                confs.append(conf)

        return (bbox, labels, confs,['openalpr_cmd'] * len(labels))
