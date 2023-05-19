import cv2 as cv
import time
import geocoder
import os
import folium
from geopy.geocoders import Nominatim

# Create a geocoder instance
geolocator = Nominatim(user_agent="detectpothole")

# Create a map
m = folium.Map(zoom_start=12)

# Function to add marker to the map
def add_marker(lat, lng):
    address = get_address(lat, lng)
    folium.Marker(location=[lat, lng], popup=address).add_to(m)

# Function to get the address from latitude and longitude coordinates
def get_address(lat, lng):
    location = geolocator.reverse((lat, lng), exactly_one=True)
    if location is not None:
        return location.address
    else:
        return "Address not found"

#reading label name from obj.names file
class_name = []
with open(os.path.join("project_files",'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

#importing model weights and config file
#defining the model parameters
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

#defining the video source (0 for camera or file name for video)
cap = cv.VideoCapture("test.mp4")
width = cap.get(3)
height = cap.get(4)
result_path = "pothole_coordinates"
os.makedirs(result_path, exist_ok=True)
fps = 10  # Desired frames per second for the compressed video
output_file = 'result_compressed.avi'

# Create a VideoWriter object for the compressed video
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter(output_file, fourcc, fps, (int(width), int(height)))

# Parameters for compression
encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 90]  # JPEG quality (0-100), higher value means better quality but larger size

#defining parameters for result saving and get coordinates
#defining initial values for some parameters in the script
g = geocoder.ip('me')
result_path = "pothole_coordinates"
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = 0

#detection loop
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if ret == False:
        break
    #analysis the stream with detection model
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w*h
        area = width*height
        #drawing detection boxes on frame for detected potholes and saving coordinates txt and photo
        if(len(scores)!=0 and scores[0]>=0.7):
            if((recarea/area)<=0.1 and box[1]<600):
                cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 1)
                cv.putText(frame, "%" + str(round(scores[0]*100,2)) + " " + label, (box[0], box[1]-10),cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
                # Get the latitude and longitude coordinates
                lat, lng = g.latlng
                # Add marker to the map
                add_marker(lat, lng)
                # Save the frame and coordinates as needed
                cv.imwrite(os.path.join(result_path, 'pothole' + str(i) + '.jpg'), frame)
                with open(os.path.join(result_path, 'pothole' + str(i) + '.txt'), 'w') as f:
                    f.write(str(lat) + ', ' + str(lng))
                i += 1
                if(i==0):
                    cv.imwrite(os.path.join(result_path,'pothole'+str(i)+'.jpg'), frame)
                    with open(os.path.join(result_path,'pothole'+str(i)+'.txt'), 'w') as f:
                        f.write(str(g.latlng))
                        i=i+1
                if(i!=0):
                    if((time.time()-b)>=2):
                        cv.imwrite(os.path.join(result_path,'pothole'+str(i)+'.jpg'), frame)
                        with open(os.path.join(result_path,'pothole'+str(i)+'.txt'), 'w') as f:
                            f.write(str(g.latlng))
                            b = time.time()
                            i = i+1

        # Writing FPS on frame
    endingTime = time.time() - starting_time
    fps = frame_counter / endingTime
    cv.putText(frame, f'FPS: {fps}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    # Showing and saving result
    cv.imshow('frame', frame)
    out.write(frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

#end
output_dir = "map"  # Specify the output directory
output_file = os.path.join(output_dir, 'map.html')  # Path to the output file
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
m.save(output_file)
cap.release()
out.release()
cv.destroyAllWindows()
