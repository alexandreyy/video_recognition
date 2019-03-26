import cv2
import tensorflow as tf
import numpy as np
import copy
import argparse
import threading
import time

from inception import inception
from inception import preprocess as inception_preprocess

import os
os.environ['QT_X11_NO_MITSHM'] = '1'        # For displaying images inside docker


X = []
info_str = ["Processing..."]
lock = threading.Semaphore()


class CallBackThread(threading.Thread):
    lock = threading.Lock()

    def __init__(self, callback, method, args, callback_args):
        self.method = method
        self.callback = callback
        self.args = args
        self.callback_args = callback_args
        super(CallBackThread, self).__init__(target=self.call)

    def call(self):
        res = self.method(self.args)
        with CallBackThread.lock:
            self.callback(res[0], res[1], self.callback_args)



class Classifier():

    def __init__ (self, rgb_inputs, rgb_logits, flow_inputs=None, flow_logits=None, classes=400, top_k=5,
                  sess=None, preprocess_rgb_fn=lambda x: x, preprocess_flow_fn=lambda x: x, fraction=1):
        self.use_flow = flow_inputs is not None and flow_logits is not None
        self.classes = [str(i) for i in range(classes)] if isinstance(classes, int) else classes
        self.top_k = min(len(self.classes), top_k)
        self.fraction = fraction

        if sess is None:
            self.sess = tf.Session()

        else:
            self.sess = sess
        
        self.rgb_inputs = rgb_inputs
        self.rgb_logits = rgb_logits
        self.predictions = self.rgb_logits
        self.preprocess_rgb_fn = preprocess_rgb_fn

        if self.use_flow:
            self.flow_inputs = flow_inputs
            self.flow_logits = flow_logits
            self.predictions += self.flow_logits
            self.preprocess_flow_fn = preprocess_flow_fn

        self.predictions = tf.nn.softmax(self.predictions)


    
    def __del__ (self):
        self.sess.close()


    def __call__(self, frames):
        X_rgb = np.expand_dims(self.preprocess_rgb_fn(frames), 0)
        feed_dict = {self.rgb_inputs: X_rgb}

        if self.use_flow:
            X_flow = np.expand_dims(self.preprocess_flow_fn(frames), 0)
        
        predictions_eval = self.sess.run([self.predictions], feed_dict=feed_dict)[0].reshape((-1,))
        indices = np.argsort(predictions_eval)[::-1][:self.top_k]

        return predictions_eval[indices], [self.classes[i] for i in indices]



class VideoClassifier(Classifier):

    def __init__ (self, rgb_inputs, rgb_logits, flow_inputs=None, flow_logits=None, classes=400, resize_fn=lambda x: x, top_k=5,
                  sess=None, num_frames=24, preprocess_rgb_fn=lambda x: x, preprocess_flow_fn=lambda x: x, fraction=1):
        super(VideoClassifier, self).__init__(rgb_inputs, rgb_logits, flow_inputs, flow_logits, classes, top_k, sess, preprocess_rgb_fn, preprocess_flow_fn, fraction)
        self.num_frames = max(num_frames, 10)
        self.resize_fn = resize_fn

    def __call__(self, input_video=-1, output_video=None, display_video=True, fps=25, parallel=True):
        video_capture = cv2.VideoCapture(input_video)
        video_capture.set(cv2.CAP_PROP_FPS, fps)
        
        if output_video:
            video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'),
                                            video_capture.get(cv2.CAP_PROP_FPS),
                                            (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                            int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        global X
        global lock

        def func():
            global X
            global lock
            
            while True:
                if len(X) >= self.num_frames:
                    X_aux = X[:self.num_frames]

                    res = super(VideoClassifier, self).__call__(X_aux)
                    VideoClassifier.callback(res[0], res[1])

                    lock.acquire()
                    X = X[len(X)-self.num_frames:]
                    #X = X[self.num_frames // self.fraction:]
                    X = X[self.num_frames - self.fraction:]
                    lock.release()
                
                time.sleep((self.num_frames - self.fraction) / 50)

        thread = threading.Thread(target=func, args=())
        thread.start()

        while True:
            available, frame = video_capture.read()

            if not available:
                break


            proc = self.resize_fn(frame)
            
            lock.acquire()
            X.append(proc)
            lock.release()
            

            # if len(X) >= self.num_frames:
            #     if self.use_flow or not parallel:
            #         predictions, classes = super(VideoClassifier, self).__call__(X)
            #         VideoClassifier.callback(predictions, classes, info_str)

            #     else:
            #         CallBackThread(callback=VideoClassifier.callback, method=super(VideoClassifier, self).__call__, args=copy.deepcopy(X), callback_args=(info_str)).start()
                
            #     X = []


            self.write_info(frame, info_str)

            if output_video is not None:
                video_writer.write(frame)
            
            if display_video:
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) & 0xff == ord('q'):
                    break
        
        video_capture.release()

        if output_video:
            video_writer.release()
        
        if len(X) >= 10:
            predictions, classes = super(VideoClassifier, self).__call__(X)
            VideoClassifier.callback(predictions, classes)


    @classmethod
    def callback(cls, predictions, classes):
        global info_str
        global lock
        lock.acquire()
        info_str.clear()
        info_str += [format(round(y*100)/100.0, '0.2f') + "   " + x for x, y in zip(classes, predictions)]
        lock.release()
        print("\n".join(info_str), "\n")

    def write_info(self, frame, info_str):
        cv2.putText(img=frame, text=info_str[0], org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(50,50,255), thickness=2)
        for i, line in enumerate(info_str[1:]):
           cv2.putText(img=frame, text=line, org=(20, 20 + (i+1)*25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255,255,0), thickness=2)

        return frame



class KineticsVideoClassifier(VideoClassifier):
    def __init__ (self, rgb_weights, flow_weights=None, top_k=5, num_frames=24, sess=None, kinetics_labels='config/kinetics_labels.txt', fraction=1):
        if sess is None:
            sess = tf.Session()

        with tf.variable_scope('inception'):
            rgb_inputs = tf.placeholder(tf.float32, shape=(1, None, 224, 224, 3))
            rgb_logits, _ = inception.load_kinetics_weights(sess, rgb_inputs, rgb_weights, 400, net_type='RGB', training=False)

            flow_inputs, flow_logits = None, None

            if flow_weights:
                flow_inputs = tf.placeholder(tf.float32, shape=(1, None, 224, 224, 2))
                flow_logits, _ = inception.load_kinetics_weights(sess, flow_inputs, flow_weights, 400, net_type='Flow', training=False)

        class_labels = [x.strip() for x in open(kinetics_labels)]
        
        super(KineticsVideoClassifier, self).__init__(rgb_inputs, rgb_logits, flow_inputs, flow_logits, class_labels, inception_preprocess.resize, 
                                                      top_k, sess, num_frames, inception_preprocess.rgb, inception_preprocess.flow, fraction)



def main():
    parser = argparse.ArgumentParser(description='Scene classification given an input video')

    parser.add_argument('-rgb', type=str,
                        help='Path to the rgb model',
                        default='../kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt')
    parser.add_argument('-flow', type=str,
                        help='Path to the optical flow model',
                        required=False)
    parser.add_argument('-frames', type=int,
                        help='Number of frames to proccess',
                        default=25)
    parser.add_argument('-i', type=str,
                        help='Input video file, or path to a camera device',
                        required=True)
    parser.add_argument('-o', type=str,
                        help='Path to the output video to be recorded',
                        required=False)
    parser.add_argument('-net', type=str,
                        help='The network architecture to use (default is inception)',
                        default='inception')
    parser.add_argument('-fraction', type=int,
                        help='Fraction of the frames to reuse',
                        default=1)

    parser.add_argument('--display', dest='display', action='store_true', help='Display the video while processing')
    parser.add_argument('--no-display', dest='display', action='store_false', help='Dont display the video while processing')
    parser.set_defaults(display=True)

    parser.add_argument('--parallel', dest='parallel', action='store_true', help='Execute the classification in parallel')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false', help='Dont execute the classification in parallel')
    parser.set_defaults(parallel=True)

    args = vars(parser.parse_args())


    if args['net'] == 'inception':
        video_classifier = KineticsVideoClassifier(rgb_weights=args['rgb'], flow_weights=args['flow'], num_frames=args['frames'], fraction=args['fraction'])

    video_classifier(input_video=args['i'], output_video=args['o'], display_video=args['display'], parallel=args['parallel'])



if __name__ == '__main__':
    main()