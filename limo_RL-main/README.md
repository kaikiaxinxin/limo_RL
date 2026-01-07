killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
cd ~/STL-Projects/limo_RL/limo_RL-main
rm -rf build/ devel/
catkin_make
source devel/setup.bash

# 终端 1
cd ~/STL-Projects/limo_RL/limo_RL-main
source devel/setup.bash
roslaunch limo_gazebo_sim limo_ackerman.launch


# 终端 2
cd ~/STL-Projects/limo_RL/limo_RL-main
source devel/setup.bash
roslaunch limoRL limo_TD3.launch

代码仓更新：
cd ~/STL-Projects/limo_RL
git status
git add .
git commit -m "xx"
git push

上传新的代码仓库：
git init
git add .
git commit -m "第一次提交代码"
git branch -M main

git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
