    def show_camie(self):
        # disabled cam, naglalag kasi
        cam = self.help.get_screen('camera').ids.camie
        self.clock_event = Clock.schedule_interval(self.object_detection, 1.0 /60)
        cam.capture = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
        self.help.current = 'camera'
        self.help.transition.direction = "left"

        
    def object_detection(self, dt):
        thres = 0.5  # Threshold to detect object
        nms_threshold = 0.2  #
        classNames = []
        with open('object-detector/coco.names', 'r') as f:
            classNames = f.read().splitlines()

        font = cv2.FONT_HERSHEY_PLAIN

        Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

        weightsPath = "object-detector/frozen_inference_graph.pb"
        configPath = "object-detector/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        cam = self.help.get_screen('camera').ids.camie
        ret, frame = cam.capture.read()  
        if ret:
            cv2.imshow('frame', frame)
            classIds, confs, bbox = net.detect(frame, confThreshold=thres)
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))
            indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

            if len(classIds) != 0:
                for i in indices:
                    # i = i[1]
                    box = bbox[i]
                    confidence = str(round(confs[i], 2))
                    color = Colors[classIds[i] - 1]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
                    cv2.putText(frame, classNames[classIds[i] - 1] + " " + confidence, (x + 10, y + 20),
                                font, 1, color, 2)
                      cv2.putText(img,str(round(confidence,2)),(box[0]+100,box[1]+30),
                                    font,1,colors[classId-1],2)
                    cv2.imshow("Output", frame)
                    cv2.waitKey(1)

            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture =Texture.create(
                size = (frame.shape[1], frame.shape[0]), colorfmt='bgr'
            )
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            cam.texture = image_texture
            self.help.get_screen('camera').ids.capture.add_widget(
                MDFloatingActionButton( 
                    text = "Capture", 
                    icon = 'camera-iris', 
                    md_bg_color="#f8d7e3", 
                    text_color = "#211c29",
                    on_press = lambda x: cv2.imwrite(f'captured_img/image.png', frame),
                    on_release = lambda x : self.capture())

    def capture(self):
        self.help.get_screen('image').ids.cap_img.source = 'captured_img/image.png'
        self.swtchScreen('image')