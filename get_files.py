#! /usr/bin/python3

import airsim
import cv2
import shutil
import time
import os
import random
import numpy as np
import glob
import math

get_width = lambda cv2_img : (cv2_img.shape[1])
get_height = lambda cv2_img : (cv2_img.shape[0])

def yolo_format(det,png):

        w = det.box2D.max.x_val - det.box2D.min.x_val
        h = det.box2D.max.y_val - det.box2D.min.y_val
        x = det.box2D.min.x_val + w // 2
        y = det.box2D.min.y_val + h // 2

        imw = get_width(png)
        imh = get_height(png)

        class_item = 0
        data = "%d %f %f %f %f" % (
            class_item,
            x / imw,
            y / imh,
            w / imw,
            h / imh
        )
        return data


def change_background(client):

    # Texture Paths 
    textures = {
        'dam_wall' : '/home/catec/Documents/unreal_iria/HDRI/dam_wall_4k.exr',
        'studio' : '/home/catec/Documents/unreal_iria/HDRI/poly_haven_studio_4k.exr',
        'street' : '/home/catec/Documents/unreal_iria/HDRI/cobblestone_street_night_4k.exr',
        'room' : '/home/catec/Documents/unreal_iria/HDRI/small_empty_room_1_4k.exr'
    }

    object_name = "Inverted_Sphere_9"

    for texture, path in list(textures.items()):
        texture_path = path                            
        ret = client.simSetObjectMaterialFromTexture(object_name, texture_path)     

        print(texture)

        time.sleep(30)

def change_ext_pose(client):
    curr_pos = client.simGetObjectPose('10285_Fire_Extinguisher_v3_iterations-2_2').position

    new_pos = airsim.Vector3r(curr_pos.x_val + random.uniform(-2,2), curr_pos.y_val + random.uniform(0,-3), curr_pos.z_val + random.uniform(-1,1))

    client.simSetObjectPose(
        '10285_Fire_Extinguisher_v3_iterations-2_2',
        airsim.Pose(new_pos, airsim.to_quaternion(np.deg2rad(random.randint(0,360)), np.deg2rad(random.randint(0,360)), np.deg2rad(random.randint(0,360)))),    # Random position and orientation
        # airsim.Pose(curr_pos, airsim.to_quaternion(np.deg2rad(random.randint(0,360)), np.deg2rad(random.randint(0,360)), np.deg2rad(random.randint(0,360)))), # Random orientation
        True
    )


if __name__ == '__main__':
    
    # Define client
    client = airsim.VehicleClient()
    client.confirmConnection()

    camera_name = "0"
    image_type = airsim.ImageType.Scene

    client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100) 
    client.simAddDetectionFilterMeshName(camera_name, image_type, "10285_Fire_Extinguisher*") 

    # Create directory to save files
    try:
        os.mkdir("Files")
    except:
        shutil.rmtree('./Files')
        os.mkdir("Files")

    
    # Set vehicle pose in sphere center
    client.simSetVehiclePose(
        client.simGetObjectPose('Inverted_Sphere_9'),
        True
    )

    # Constants
    cont = 0
    r = 2.9 # Sphere radius [m]

    try:
        while True:
            if not client.simIsPause():
                rawImage = client.simGetImage(camera_name, image_type)
                if not rawImage:
                    exit()

                print(cont)

                ## Save files
                png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
                detects = client.simGetDetections(camera_name, image_type)
                
                if detects:
                    with open(f'Files/exting{cont}.txt','w') as f:
                        for detect in detects:
                            data = yolo_format(detect,png)
                            f.write(data)
                
                cv2.imwrite(f'Files/exting{cont}.jpg',png)
                
                ## Change background
                client.simSetObjectMaterialFromTexture(
                    'Inverted_Sphere_9',
                    random.choice(glob.glob('/home/catec/Documents/unreal_iria/backgrounds/*'))
                )

                ## Change camera pose
                client.simSetVehiclePose(
                    airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(np.deg2rad(-20), 0, -cont*math.pi/4)),
                    True
                )

                veh_pose = client.simGetVehiclePose()


                [b,y,a] = airsim.utils.to_eularian_angles(veh_pose.orientation)

                rm  = [
                    [ np.cos(a) * np.cos(b) * np.cos(y) - np.sin(a) * np.sin(y),     -np.cos(a) * np.cos(b) * np.sin(y) - np.sin(a) * np.cos(y),     np.cos(a) * np.sin(b),  veh_pose.position.x_val],
                    [ np.sin(a) * np.cos(b) * np.cos(y) + np.cos(a) * np.sin(y),     -np.sin(a) * np.cos(b) * np.sin(y) + np.cos(a) * np.cos(y),     np.sin(a) * np.sin(b),  veh_pose.position.y_val],
                    [-np.sin(b) * np.cos(y)                                    ,      np.sin(b) * np.sin(y)                                    ,     np.cos(b)            ,  veh_pose.position.z_val],
                    [ 0,0,0,1],
                ]

                pos_rel = np.array([2,0,0.5,1])   # frente/izq/abajo

                pos = np.matmul(rm,pos_rel)

                client.simSetObjectPose(
                    '10285_Fire_Extinguisher_v3_iterations-2_2',
                    airsim.Pose(airsim.Vector3r(pos[0],pos[1],pos[2]), airsim.to_quaternion(np.deg2rad(random.randint(0,360)), np.deg2rad(random.randint(0,360)), np.deg2rad(random.randint(0,360)))),   
                    True
                    )

                time.sleep(0.1)


                cont +=1
            else:
                pass
    except KeyboardInterrupt: 

        destination = "/home/catec/Documents/unreal_iria/Extinguisher/extinguisher"

        try:
            os.mkdir(destination + "/train")
            os.mkdir(destination + "/train/images")
            os.mkdir(destination + "/train/labels")

            os.mkdir(destination + "/valid")
            os.mkdir(destination + "/valid/images")
            os.mkdir(destination + "/valid/labels")
        except:
            shutil.rmtree('./extinguisher/train')
            shutil.rmtree('./extinguisher/valid')

            os.mkdir(destination + "/train")
            os.mkdir(destination + "/train/images")
            os.mkdir(destination + "/train/labels")

            os.mkdir(destination + "/valid")
            os.mkdir(destination + "/valid/images")
            os.mkdir(destination + "/valid/labels")

        
        dirs = os.listdir('./Files')
        length = math.trunc(len(dirs)/2)+1

        for file in dirs:
            for i in range(length):
                if i<=(length*0.75) and file.endswith(f'g{i}.jpg'):
                    shutil.copy(f"./Files/{file}", destination + "/train/images")

                elif i<=(length*0.75) and file.endswith(f'g{i}.txt'):
                    shutil.copy(f"./Files/{file}", destination + "/train/labels")

                elif i>(length*0.75) and file.endswith(f'g{i}.jpg'):
                    shutil.copy(f"./Files/{file}", destination + "/valid/images")

                elif i>(length*0.75) and file.endswith(f'g{i}.txt'):
                    shutil.copy(f"./Files/{file}", destination + "/valid/labels")








    
