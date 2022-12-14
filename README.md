# SLAM for THOR
SLAM using Particle Filtering for THOR Humanoid Robot

Simultaneous Localization and Mapping of the Humanoid THOR robot is done using IMU data and 2-D LiDAR scans. The LiDAR scans are transformed into an occupancy grid map and the robot pose is initialized with a number of particles. Using log-odds, the weight of each particle is updated. Then stratified resampling is done to sample a new set of particles and update their poses and weights iteratively to ultimately generate a map of the unknown environment.

In the following maps, `cyan` represents the trajectory generated by odometry data and `red` represents the filtered trajectory.
## Occupancy Maps
Map 1 <br>
<img src="https://user-images.githubusercontent.com/46754269/204720432-762e5d27-0832-4477-8c9e-d9297f9e181a.png" width="300" height="300"> <br>
Map 2 <br>
<img src="https://user-images.githubusercontent.com/46754269/204720713-8738b677-4f3a-41fc-a7a7-a2dce9be85bc.png" width="300" height="300"> <br>
Map 3 <br>
<img src="https://user-images.githubusercontent.com/46754269/204720737-400a456a-64a8-44ec-be78-354deaf91119.png" width="300" height="300"> <br>
Map 4 <br>
<img src="https://user-images.githubusercontent.com/46754269/204720754-e9eb7151-ffbe-4b29-86c6-dadee2b363dc.png" width="300" height="300">
