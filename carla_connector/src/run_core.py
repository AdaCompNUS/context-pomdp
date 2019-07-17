#!/usr/bin/env python


import rospy

from roslaunch.parent import ROSLaunchParent

import subprocess

if __name__ == '__main__':

    try:
        # carla_path = sys.path.expanduser("~/carla/")
        # subprocess.Popen("./CarlaUE4.sh",cwd=carla_path, shell=True)

        parent = ROSLaunchParent("mycore", [], is_core=True)     # run_id can be any string
        parent.start()


        rospy.init_node('dummy')

       
        rospy.spin()
        # connector.loop()

    finally:
        print("Terminating...")
        
        parent.shutdown()


    
     

