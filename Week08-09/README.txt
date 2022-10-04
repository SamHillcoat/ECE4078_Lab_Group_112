Start Gazebo
Add map to Gazebo (M$_True_Map)
run python3 gui.py
press \ (backslash) until a suitable path comes up (look for one with lots of straight lines and no overlap)
press enter to start driving, gui will switch to slam view


Reading console output:
The console will output the robot state (from slam) at each iteration of control loop
when in turning loop it will also output the heading error which it being minimized.

If the robot appears to be stopped it is not necessarily broken just wait for it to turn very slowly.
Output also shows wheel velocities so if these are not 0 then it is just really slow.