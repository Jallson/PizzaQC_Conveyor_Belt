'''
	Author: Jallson Suryo & Nicholas Patrick
	Date: 2023-12-15
	License: CC0
	Source: Edge Impulse python example file classify.py -- modified
	Description: Program to detect whether the pizza toppings on the moving conveyer belt are the correct amount. Will give OK or Bad output for each pizza that passes.
'''
#!/usr/bin/env python

import device_patches       # Device specific patches for Jetson Nano (needs to be before importing cv2)

import cv2
import os
import sys, getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner

runner = None
# if you don't want to see a camera preview, set this to False
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

def now():
    return round(time.time() * 1000)

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" %port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName =camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']
            if len(args)>= 2:
                videoCaptureDeviceId = int(args[1])
            else:
                port_ids = get_webcams()
                if len(port_ids) == 0:
                    raise Exception('Cannot find any webcams')
                if len(args)<= 1 and len(port_ids)> 1:
                    raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
                videoCaptureDeviceId = int(port_ids[0])

            camera = cv2.VideoCapture(videoCaptureDeviceId)
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
                camera.release()
            else:
                raise Exception("Couldn't initialize selected camera.")

            next_frame = 0 # limit to ~10 fps here

            # topping counting helper variables
            topping_names = ["mush", "papri", "roni"]
            topping_good_count = [3, 3, 3]
            topping_prev_count = [0] * len(topping_names)
            movingIn = True

            for res, img in runner.classifier(videoCaptureDeviceId):
                if (next_frame > now()):
                    time.sleep((next_frame - now()) / 1000)

                # print('classification runner response', res)

                if "classification" in res["result"].keys():
                    print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                    for label in labels:
                        score = res['result']['classification'][label]
                        print('%s: %.2f\t' % (label, score), end='')
                    print('', flush=True)

                elif "bounding_boxes" in res["result"].keys():
                    # count the occurrence of each topping in this frame
                    topping_real_count = [0] * len(topping_names)
                    for bb in res["result"]["bounding_boxes"]:
                        topping_real_count[topping_names.index(bb['label'])] += 1

                    # determine if the pizza is currently moving out or moving in
                    switch = False
                    for i in range(len(topping_names)):
                        if movingIn:
                            # if any topping disappeared, it's moving out
                            if topping_prev_count[i] > topping_real_count[i]:
                                switch = True
                                break
                        else:
                            # if there are no toppings, it's moving in
                            if sum(topping_real_count) == 0:
                                switch = True
                                break
                    report = movingIn and switch
                    movingIn = movingIn ^ switch

                    # report the results if the pizza was moving in and now moving out
                    if report:
                        toPrint = ""
                        if topping_prev_count == topping_good_count: toPrint += "Ok:"
                        else: toPrint += "Bad:"
                        for i in range(len(topping_names)):
                            toPrint += " %s: %d" % (topping_names[i], topping_prev_count[i])
                            if i + 1 < len(topping_names): toPrint += ","
                        print(toPrint)

                    # set the previous count to be the current count
                    topping_prev_count = topping_real_count

                if (show_camera):
                    cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) == ord('q'):
                        break

                next_frame = now() + 100
        finally:
            if (runner):
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])
