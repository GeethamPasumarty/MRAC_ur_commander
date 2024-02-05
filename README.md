### IP_Mapper

### Setup

1. Build the image
    1. git clone this repo
    ```bash
    cd MRAC_ur_commander

    ```
    2. Build and run the image
    ```bash
    git pull
    .docker/build_image.sh
    .docker/run_user.sh
    sudo chown -R $user /dev_ws
    ```
2. Start the moveit
    1. terminator
    2. Connect the robot
        1. connect the LAN to your PC
        2. change the IP address to 192.168.56.1, mask 24
    3. Launch the Moveit commander and camera (split terminal)
        ```bash
        roslaunch commander ur10e_ka_commander.launch
        roslaunch capture_manager ka_capture_manager.launch
        ```
    4. Create a new notebook for scanning
       1. First, test it in a simulation
       2. add the capture script (save the rgb_image and depth_imge. You need to make a new direct subscribe to the topic, which are rgb_raw and depth_img_raw, then save it to the image folder)
       3. test the height of the camera(you need have different definition of the TCP and find the best one for scanning)
    
3. Image processing
    1. Edit the file location in the "Wood Detection" file.
    2. Extract the required outputs.
    3. The next step will be to move it to a rosnode
    4. to find the x,y position. you need to get the x,y coord of the pixel, then extract the depth of the point and inverse calculation from the RGB camera link. (world x,y,z)

4. GH and ros(song)
    1. down the rosbridge and receive the coordinate from Ros (this should be done in Grasshopper)
    2. create a digital twin of the ur robot and abb robot
    3. finish the simulation
    4. gh_ros link https://github.com/HuanyuL/gh_ros go the option one(compass_fab)

5. push the changes
    ```bash
    git add .
    git commit -m 'your message'
    git push
    ```

