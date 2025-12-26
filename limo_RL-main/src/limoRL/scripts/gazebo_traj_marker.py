#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose

class TrajectoryMarker(object):
    def __init__(self):
        rospy.init_node('trajectory_marker')
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.idx = 0
        self.last_pose = None
        # 修改为你自己的 marker_box 路径
        self.marker_path = '/home/robot/RL/limo_RL-main/src/limoRL/models/marker_box/model.sdf'

    def odom_callback(self, msg):
        pose = msg.pose.pose
        if self.last_pose is not None:
            dx = pose.position.x - self.last_pose.position.x
            dy = pose.position.y - self.last_pose.position.y
            if (dx**2 + dy**2) < 0.05**2:
                return
        self.last_pose = pose

        with open(self.marker_path, 'r') as f:
            model_xml = f.read()
        model_name = "marker_%d" % self.idx
        self.idx += 1

        marker_pose = Pose()
        marker_pose.position.x = pose.position.x
        marker_pose.position.y = pose.position.y
        marker_pose.position.z = 0.0001
        marker_pose.orientation = pose.orientation

        try:
            self.spawn_model(model_name, model_xml, '', marker_pose, 'world')
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to spawn marker: %s" % e)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = TrajectoryMarker()
    node.run()