from pyzm import __version__ as pyzmversion
import pyzm.api as zmapi
import getpass
import traceback
import pyzm.ZMMemory as zmmemory
import time
#import pyzm.ml.object as  ObjectDetect
from pyzm.ml.detect_sequence import DetectSequence
from pyzm.helpers.Base import ConsoleLog
import pyzm.helpers.utils as utils
import sys

#import pyzm.ZMLog as log 


print ('Using pyzm version: {}'.format(pyzmversion))

#log.init(name='stream', override={'dump_console': True})
#logger = log
logger = ConsoleLog()
logger.set_level(5)

#time.sleep(1000)

if len(sys.argv) == 1:
    eid = input ('Enter event ID to analyze:')
else:
    eid = sys.argv[1]



'''
api_options = {
    'apiurl': 'https://demo.zoneminder.com/zm/api',
    'portalurl': 'https://demo.zoneminder.com/zm',
    'user': 'zmuser',
    'password': 'zmpass',
    'logger': logger, # use none if you don't want to log to ZM,
    #'disable_ssl_cert_check': True
}
'''

conf = utils.read_config('/etc/zm/secrets.ini')
api_options  = {
    'apiurl': utils.get(key='ZM_API_PORTAL', section='secrets', conf=conf),
    'portalurl':utils.get(key='ZM_PORTAL', section='secrets', conf=conf),
    'user': utils.get(key='ZM_USER', section='secrets', conf=conf),
    'password': utils.get(key='ZM_PASSWORD', section='secrets', conf=conf),
    'logger': logger, # use none if you don't want to log to ZM,
    #'disable_ssl_cert_check': True
}


zmapi = zmapi.ZMApi(options=api_options, logger=logger)
ml_options = {
    'general': {
        'model_sequence': 'object,face,alpr',
        'disable_locks': 'no'

    },
   
    'object': {
        'general':{
            'pattern':'car',
            'same_model_sequence_strategy': 'most' # also 'most', 'most_unique's
        },
        'sequence': [{
            #First run on TPU
            'object_weights':'/var/lib/zmeventnotification/models/coral_edgetpu/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
            'object_labels': '/var/lib/zmeventnotification/models/coral_edgetpu/coco_indexed.names',
            'object_min_confidence': 0.3,
            'object_framework':'coral_edgetpu'
        },
        {
            # YoloV4 on GPU if TPU fails (because sequence strategy is 'first')
            'object_config':'/var/lib/zmeventnotification/models/yolov4/yolov4.cfg',
            'object_weights':'/var/lib/zmeventnotification/models/yolov4/yolov4.weights',
            'object_labels': '/var/lib/zmeventnotification/models/yolov4/coco.names',
            'object_min_confidence': 0.3,
            'object_framework':'opencv',
            'object_processor': 'cpu',
            #'model_width': 512,
            #'model_height': 512
        }]
    },
    'face': {
        'general':{
            'pattern': '.*',
            'same_model_sequence_strategy': 'first'
        },
        'sequence': [{
            'face_detection_framework': 'dlib',
            'known_images_path': '/var/lib/zmeventnotification/known_faces',
            'face_model': 'cnn',
            'face_train_model': 'cnn',
            'face_recog_dist_threshold': 0.6,
            'face_num_jitters': 1,
            'face_upsample_times':1,
            'max_size': 800
        }]
    },

    'alpr': {
         'general':{
            'same_model_sequence_strategy': 'first',
            'pre_existing_labels':['car', 'motorbike', 'bus', 'truck', 'boat'],

        },
         'sequence': [{
            'alpr_api_type': 'cloud',
            'alpr_service': 'plate_recognizer',
            'alpr_key': utils.get(key='PLATEREC_ALPR_KEY', section='secrets', conf=conf),
            'platrec_stats': 'no',
            'platerec_min_dscore': 0.1,
            'platerec_min_score': 0.2,
         }]
    }
} # ml_options

stream_options = {
        #'frame_skip':2,
        #'start_frame': 21,
        #'max_frames':20,
        'strategy': 'most_models',
        #'strategy': 'first',
        'api': zmapi,
        'download': False,
        'frame_set': 'snapshot,alarm,7000,25,35,45',
        'resize': 800,
        'save_frames': False,
        'save_analyzed_frames': False,
        'save_frames_dir': '/tmp',
        'contig_frames_before_error': 5,
        'max_attempts': 3,
        'sleep_between_attempts': 4,
        'disable_ssl_cert_check': True
}


#input ('Enter...')
m = DetectSequence(options=ml_options, logger=logger)
#m = ObjectDetect.Object(options=ml_options)
matched_data,all_data = m.detect_stream(stream=eid, options=stream_options)
print(f'ALL FRAMES: {all_data}\n\n')
print (f"SELECTED FRAME {matched_data['frame_id']}, size {matched_data['image_dimensions']} with LABELS {matched_data['labels']} {matched_data['boxes']} {matched_data['confidences']}")

