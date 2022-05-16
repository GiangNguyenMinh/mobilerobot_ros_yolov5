# Mobile robot use deep_sort + Yolov5
## Install 
Clone from repo
```bash
git clone --recurse-submodules https://github.com/GiangNguyenMinh/mobilerobot_ros_yolov5.git
cd [your catkin ws directory]/src
cp -r ~/mobilerobot_ros_yolov5/m2wr_description .
```
Create env for yolov5+deepsort
```
cd ~/[your catkin ws directory] 
python3 -m venv env
source env/bin/activate
cd src/m2wr_description/scripts
pip install -r requirement.txt
cd ~/[your catkin ws directory] 
catkin_make
```
Clone Clone deep_sort_pytorch weight from [here](https://drive.google.com/file/d/1WrAPyakHLov2-h_FRahsbH4IWFaFt572/view?usp=sharing), then copy to "src/m2wr_description/scripts/deep_sort_pytorch/deep_sort/deep/checkpoint"
# Run
Open new terminator (Ctrl + Alt + T)
```bash
roscore
```
Open new terminator
```bash
cd [your catkin ws directory] 
source env/bin/activate
source devel/setup.bash
roslaunch m2wr_description spawn.launch
```
Open new terminator
```bash
cd [your catkin ws directory] 
source env/bin/activate
source devel/setup.bash
cd src/m2wr_description/scripts
rosrun m2wr_description ros_cam.py
```



