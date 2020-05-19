
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame

    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        Loading the network in core
        '''
        self.plugin = IECore()
        self.exec_network = self.plugin.load_network(network=self.model, device_name=self.device)

    def predict(self, image):
        '''
        Detecting people in image
        '''
        preprocessed_input = self.preprocess_input(image)

        self.exec_network.infer({self.input_name:preprocessed_input})

        result = self.exec_network.requests[0]

        coords = self.preprocess_outputs(result.outputs['detection_out'])
        height, width = image.shape[:2]
        for coord in coords:
            coord[0] = coord[0] * width
            coord[1] = coord[1] * height
            coord[2] = coord[2] * width
            coord[3] = coord[3] * height

        preprocessed_image = self.draw_outputs(coords, image)
        return coords, preprocessed_image

    def draw_outputs(self, coords, image):
        '''
        Drawing rectangles around detected people
        '''
        for coord in coords:
            (startX, startY, endX, endY) = coord
            cv2.rectangle(image, (startX, startY), (endX, endY), (255,0,0), 2)
            cv2.rectangle(image, (620,1), (915, 562), (0,0,0), 5)

        return image

    def preprocess_outputs(self, outputs):
        '''
        Processing the output to get the bounding box with required threshold
        '''
        coords = []
        for i in np.arange(0, outputs.shape[2]):
            confidence = outputs[0,0,i,2]
            if confidence > self.threshold:
                box = outputs[0, 0, i, 3:7]
                coords.append(box)
        return coords

    def preprocess_input(self, image):
        '''
        Preprocessing the input to fit the the inference engine
        '''
        b, c, h, w = self.input_shape
        prepo = np.copy(image)
        prepo = cv2.resize(prepo, (w,h))
        prepo = prepo.transpose((2,0,1))
        prepo = prepo.reshape(1,c,h,w)
        return prepo

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    counter=0
    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1

            coords, image= pd.predict(frame)
            print(f"Total People in frame = {len(coords)}")
            y_pixel=25

            out_video.write(image)

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)

    args=parser.parse_args()

    main(args)
