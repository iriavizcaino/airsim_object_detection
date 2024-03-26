import airsim
import transforms3d
import numpy as np
import random
import time

if __name__ == '__main__':
    
    # Define clients
    client = airsim.VehicleClient()
    client.confirmConnection()

    camera_name = "0"
    image_type = airsim.ImageType.Scene

    client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100) 
    client.simAddDetectionFilterMeshName(camera_name, image_type, "10285_Fire_Extinguisher*") 


    while True:

        veh_pose = client.simGetVehiclePose()

        [b,y,a] = airsim.utils.to_eularian_angles(veh_pose.orientation)

        rm  = [
            [ np.cos(a) * np.cos(b) * np.cos(y) - np.sin(a) * np.sin(y),     -np.cos(a) * np.cos(b) * np.sin(y) - np.sin(a) * np.cos(y),     np.cos(a) * np.sin(b),  veh_pose.position.x_val],
            [ np.sin(a) * np.cos(b) * np.cos(y) + np.cos(a) * np.sin(y),     -np.sin(a) * np.cos(b) * np.sin(y) + np.cos(a) * np.cos(y),     np.sin(a) * np.sin(b),  veh_pose.position.y_val],
            [-np.sin(b) * np.cos(y)                                    ,      np.sin(b) * np.sin(y)                                    ,     np.cos(b)            ,  veh_pose.position.z_val],
            [0,0,0,1],
        ]

        pos_rel = np.array([4,0,0,1])

        pos = np.matmul(rm,pos_rel)

        client.simSetObjectPose(
            '10285_Fire_Extinguisher_v3_iterations-2_2',
            airsim.Pose(airsim.Vector3r(pos[0],pos[1],pos[2]), airsim.to_quaternion(np.deg2rad(random.randint(0,360)), np.deg2rad(random.randint(0,360)), np.deg2rad(random.randint(0,360)))),   
            True
            )
        
        # time.sleep(0.1)
    
    




    
