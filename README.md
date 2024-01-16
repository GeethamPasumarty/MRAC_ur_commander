### Project Descrption/Name

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
        1. connect the lan to you pc
        2. change the ip address to 192.168.56.1, mask 24
    3. Launch the moveit commander and camera (split terminal)
        ```bash
        roslaunch commander ur10e_ka_commander.launch
        roslaunch capture_manager ka_capture_manager.launch
        ```
    4. Create a new notebook for scanning(Sai)
       1. first test it in simulation
       2. add the capture script (save the rgb_image and depth_imge. You need to make a new direct subscribe to the topic which are rgb_raw and depth_img_raw then save it to the image folder)
       3. test the height of the camera(you need have different definition of the tcp and find the best one for scanning)
    
3. image processing(geetham)
    1. finish and test the image from the test
    2. next step will be move it to a rosnode
    3. to find the x,y position. you need to get the x,y coord of the pixel then extracrt the depth of the point and inverse calculation from the rgb camera link. (world x,y,z)

4. GH and ros(song)
    1. down the rosbridge and recevive the coordinate from ros(this should be done in grasshopper)
    2. create a digital twin of the ur robot and abb robot
    3. finish the simulation
    4. gh_ros link https://github.com/HuanyuL/gh_ros go the option one(compass_fab)

5. push the changes
    ```bash
    git add .
    git commit -m 'your message'
    git push
    ```

