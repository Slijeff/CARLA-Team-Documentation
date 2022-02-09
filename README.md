# CARLA Sensing Team Documentation

---

## This is a Sensing Group Documentation for semester 2021 Fall by Zhaoyang Li, Jeffrey, Yizhou Chen

## Part A: Theory

---

### **Object Recognition (Object Detection)**

O*bject recognition (object detection) was combined with two computer vision tasks which are object classification and object localization.* 

*Object Classification: Predict the type or class of an object in an image, in which we input an image with a single object and output a class label.*

Object *localization: Locate the objects in an image and indicate their location with a bounding box, in which we input an image with one or more objects and output one or more bounding boxes.*

*Object Detection: Locate the presence of objects with a bounding box and types or classes of the located objects in an image. in which we input an image with one or more objects nad output one or more bounding boxes and a class label for each bounding box.*

### Fast-RCNN

![Untitled](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/Untitled.png)

1. We convolve and pool the image once, and at the same time, we also do a selective search (SS) which checks the existing small areas, merge the two most likely areas, repeat this step until the image is merged into one area, and finally output the candidate area. And then we get the feature map from the convolution and polling and region proposal from the SS, and we use the feature map and region proposal to do region-of-interest pooling (ROI) which *Refers to the mapping of the “selective box" on the feature map obtained after the SS is completed. And we use FCN to classify and get the score of each class.*
2. *Faster-RCNN: The four steps are all handed over to the deep neural network to do, and all run on the GPU, which greatly improves the efficiency of the operation.*

### YOLO

![Untitled](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/Untitled%201.png)

Overall, the Yolo algorithm uses a single CNN model to achieve end-to-end target detection. The entire system is shown in Figure 5. First, resize the input image to 448x448, then send it to the CNN network, and finally process the network prediction results. The target of detection. Compared with the R-CNN algorithm, it is a unified framework, and its speed is faster, and the training process of Yolo is also end-to-end.

After removing the candidate area, the structure of YOLO is very simple, that is, pure convolution and pooling, and finally, two layers of full connections are added. The biggest difference is that the final output layer uses a linear function as the activation function because it needs to predict the position of the bounding box (numerical type), not just the probability of the object. The YOLO network structure is composed of 24 convolutional layers and 2 fully connected layers. The network entrance is 448x448 (v2 is 416x416). The picture enters the network and undergoes resize. The output result of the network is a tensor with the dimensions: $S * S * (B * 5 + C)$, where S is the number of divided grids, B is the number of frames each grid is responsible for, and C is the number of categories. Each small grid corresponds to B bounding boxes, and the width and height range of the bounding box is the full image, which means that the location of the bounding box of the object is searched for with the small grid as the center. Represents whether there are objects in the place and the positioning accuracy: $P(objec)*IoU^{truth}_{pred}$, Each small grid will correspond to C probability values, find the category P(Class|object) corresponding to the maximum probability, and consider that the small grid contains the object or part of the object.

## Part B: Conclusion for mainly used CARLA API

---

### Gnerate Vehicle

```python
client = carla.Client('localhost', 2000)  # https://carla.readthedocs.io/en/0.9.11/core_world/#the-client
client.set_timeout(10.0)

world = client.get_world()

blueprint_library = world.get_blueprint_library()
vehicle = blueprint_library.find('vehicle.tesla.model3')
sensor = blueprint_library.find('sensor.camera.rgb')

r_bp = blueprint_library.find('static.prop.colacan')

# spawn points
spawn_points = world.get_map().get_spawn_points()[10]
# change the dimensions of the image
sensor.set_attribute('image_size_x', f'{IM_WIDTH}')
sensor.set_attribute('image_size_y', f'{IM_HEIGHT}')
sensor.set_attribute('fov', '110')
# spawn vehicle
actor_vehicle = world.spawn_actor(blueprint=vehicle, transform=spawn_points)
# get the relative coordinates of the created car
spawn_point_car = carla.Transform(carla.Location(x=0.6, y=-0.45, z=1.6))
# spawn the sensor
actor_sensor = world.spawn_actor(blueprint=sensor, transform=spawn_point_car,attach_to=actor_vehicle)
```

[carla.mp4](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/carla.mp4)

### Visualization Data

[https://www.notion.so](https://www.notion.so)

```python
# get real-time points from the vehicle
actor_sensor.listen(lambda data: process_img(data, "sensor2"))
actor_vehicle.set_autopilot(True)
x_list_vehicle = []
y_list_vehicle = []
z_list_vehicle = []
x_list_senor = []
y_list_senor = []
z_list_senor = []

for i in range(20):
    actor_vehicle_get = actor_vehicle.get_transform()
    actor_sensor_get = actor_sensor.get_transform()
    coordinate_vehicle_str = "(x,y,z) = ({},{},{})".format(actor_vehicle_get.location.x, actor_vehicle_get.location.y,
                                                           actor_vehicle_get.location.z)
    coordinate_sensor__str = "(x,y,z) = ({},{},{})".format(actor_sensor_get.location.x, actor_sensor_get.location.y,
                                                           actor_sensor_get.location.z)
    # put real-time coordinates in lists
    x_list_vehicle.append(actor_vehicle_get.location.x)
    y_list_vehicle.append(actor_vehicle_get.location.y)
    z_list_vehicle.append(actor_vehicle_get.location.z)
    x_list_senor.append(actor_sensor_get.location.x)
    y_list_senor.append(actor_sensor_get.location.y)
    z_list_senor.append(actor_sensor_get.location.z)
    # sleep each 5 seconds
    time.sleep(5)

# draw the plot of  real-time coordinates
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.set_title('vehicle')
plot1 = ax1.plot(x_list_vehicle, y_list_vehicle, z_list_vehicle, label="vehicle")
plt.savefig("vehicle.png")

plt.clf()
```

![sensor.png](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/sensor.png)

![vehicle.png](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/vehicle.png)

### **Randomly Pedestrian Objects and Take a Photo**

```python
actors = []

client = carla.Client('localhost', 2000)  # https://carla.readthedocs.io/en/0.9.11/core_world/#the-client
client.set_timeout(10.0)

world = client.get_world()
blueprint_library = world.get_blueprint_library()

# get pedestrian, rgb camara, and walker
walker_controller_bp = blueprint_library.find('controller.ai.walker')
walker_bp = blueprint_library.filter("walker.pedestrian.*")
sensor = blueprint_library.find('sensor.camera.rgb')
# get location of the world
trans = carla.Transform()
trans.location = world.get_random_location_from_navigation()
trans.location.z += 1
# spawn walker
walker = random.choice(walker_bp)
actor = world.spawn_actor(walker, trans)
world.wait_for_tick()
# spawn controller
controller = world.spawn_actor(walker_controller_bp, carla.Transform(), actor)
world.wait_for_tick()
# spawn rgb camara to the pedestrian
spawn_point_walker = carla.Transform(carla.Location(x=0.6, y=-0.45, z=1.6))
actor_sensor = world.spawn_actor(blueprint=sensor, transform=spawn_point_walker,
                                 attach_to=actor)
# register listener for rgb camara
actor_sensor.listen(lambda data: process_img(data, "sensor2"))

actors.append(actor)
actors.append(controller)

while True:
    time.sleep(2)
```

![Untitled](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/Untitled%202.png)

## Part C: Real-time object detection using CARLA & yolov4 with Deepsort

---

We decided to use yolov4 after trying some other object detection models, as it can perform well at good accuracy with pretrained model and real life video streams of pedestrians&cars.

Our Real-time object detection using CARLA & yolov4 with Deepsort refers to [Object-Detection-and-Tracking/OneStage/yolo at master · yehengchen/Object-Detection-and-Tracking (github.com)](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo)

Testing environment: CPU: intel i7 -10750H @2.60GHZ

### Make the repo work: changes made to make it work at our current library versions

The first thing we need to do is to change function calls and imports to ensure compatibility of yolov4&deepsort with our current libraries installed

Video input has generally good accuracy:

![Picture1.png](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/Picture1.png)

### Implementation of CARLA real-time car&people detection/tracking based on yolov4&deepsort

 Each Carla sensor in Carla has a single thread with a buffer/queue which is to buffer each frame in order into a single buffer

 Carla thread(s) → Sensor(s) → listening to extract real time image matrix/frame and append it to the buffer

 We have one YOLOv4 thread which would wait until latest image can be extracted from each buffer so that the image can be processed.

YOLOv4 thread → waiting for data extracted by Carla thread → preprocess image by scaled frame → inference of frame(classification → tracking)→ show results as video(show by openCV)/txt/recorded file(written by openCV)/..Preprocessing frame step: Most OpenCV functions require calculations based on continuous array(stored in memory)

The ascontiguousarray function converts an array that is not continuously stored in memory to an array that is continuously stored in memory, which makes the operation faster.

See reference[2] for more details about preprocessing with ascontiguousarray

Code uploaded:

[https://github.com/ychen884/Carla-YOLO4-tracking/blob/main/Object-Detection-and-Tracking-master/OneStage/yolo/deep_sort_yolov4/main.py](https://github.com/ychen884/Carla-YOLO4-tracking/blob/main/Object-Detection-and-Tracking-master/OneStage/yolo/deep_sort_yolov4/main.py)

### Real time detection&sort performance

Check screenshots below for a simple example(the performance is similar to the case for multiple cars&people): 

Red box: tracking box, record object index(here is 1)

Red dot at center of the box: tracking object movements for last x time period

White box: detection box, the disparity between boxes positions

Left window is CARLA simulator

Middle window is Carla real-time detection output

Right window is Carla RGB sensor real-time frame output 

### Why there is inaccuracy/delay of red box?

< inaccurate inference of deepsort → skipping frames to keep “real time”

Input frame: 1, 2, 3, 4, 5, 6 …    input rate ~ 10+frames  per second

Inferenced frame: 1, 20, 40, ...  inferenced frame ~ 1 frame per second, only analyze the latest frame caught by camera

Keep it “Real Time” >> skip frames 2~19, 21~39, …., which cause inaccurate tracking box

![Picture2.png](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/Picture2.png)

![Picture4.png](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/Picture4.png)

Another similar example

![Picture3.png](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/Picture3.png)

### Analyze performance & accuracy:

- Low FPS <- low inference rate
- Possible solutions with priority:
- 0. Using better GPU to run the program: Zhaoyang tried with RTX 3060(power limitation on laptop)
- FPS~1, still a problem; Ordered RTX 3090
- 1. Using smaller neural networks, such as yolov4/v3 tiny <- Current best choice before having a better hardware, and we are currently working on this
- 2. Using smaller frame size, less features -> higher loss
- 3. Replacing deepsort with self-designed re-identification algorithm relying on detection accuracy <- May not need, as this part doesn’t take too much time..

## Part D: Label & collect Carla training dataset

---

### Collecting:

### Labeling:

With the help of open source image labeling tools, we were able to efficiently label objects within images.

The tool we were using is called labelImg: https://github.com/tzutalin/labelImg

![Interface of labelImg](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/Untitled%203.png)

Interface of labelImg

![Format of each label .xml file](CARLA%20Sensing%20Team%20Documentation%2033b19946e1fb47fdac85997acc45c81d/Untitled%204.png)

Format of each label .xml file

LabelImg generates a `.xml` file for each image we collected as shown above. For training, we also need to convert these `.xml` into `.txt` for yolov4 by using a Python script.

## Part E: Training

---

After seeing how powerful Yolo detection is, we decided to train our own yolov4 model based on CARLA dataset we collected and labeled. 

In general, we followed the instructions on this GitHub repo: 

[Object-Detection-and-Tracking/OneStage/yolo/Train-a-YOLOv4-model at master · yehengchen/Object-Detection-and-Tracking](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo/Train-a-YOLOv4-model)

There are three main steps to training:

1. We downloaded the darknet source code from `[https://github.com/AlexeyAB/darknet.git](https://github.com/AlexeyAB/darknet.git)` and tried to `make` the Makefile. However, the Makefile contains Linux instructions that cannot be executed under Windows environment.
2. Next, four `.txt` files were generated using the provided tool in the repo:
    1. `train.txt` ⇒ Store all train_img name without .jpg
    2. `val.txt` ⇒ Store all val_img name without .jpg
    3. `object_train.txt` ⇒ Store all train_img **absolute path**
    4. `object_val.txt` ⇒ Store all val_img **absolute path**
    
    Since these files contains absolute paths, it was also infeasible to use Google Collab since Google Drive cannot handle absolute paths.
    

1. Create or modify `.names` `.data` and `.cfg` files
    1. `.names` file contains a list of classes to train for
    2. `.data` file contains some general training setups
    3. `.cfg` file contains the configuration of the training network itself, modified according to tutorial.
    

```python
classes = [number of objects]
train = [object_train.txt absolute path]
valid = [object_val.txt absolute path]
names = [train.names absolute path]
backup = backup/ #save weights files here
```

<aside>
⚠️ This is as far as we got. Due to the training executable cannot be make and the path issue, further training cannot be proceed. However, for the last step, we only need to run one command, which is fairly easy if we had these steps before set up.

</aside>

## Reference:

---

Low FPS <- low inference rate
Possible solutions with priority:
-0. Using better GPU to run the program: Zhaoyang tried with RTX 3060(power limitation on laptop)
FPS~1, still a problem; Ordered RTX 3090
-1. Using smaller neural networks, such as yolov4/v3 tiny  <- Current best choice before having a better hardware, and we are currently working on this
-2. Using smaller frame size, less features -> higher loss
-3. Replacing deepsort with self-designed re-identification algorithm relying on detection accuracy <- May not need, as this part doesn’t take too much time..
