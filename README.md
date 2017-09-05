# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
In this exercise many filters are applied to the incoming Point cloud (pcl) image. Each filter has a job, which in conjunction, remove noise from the image as well as separate the objects on the table from the table itself. 

#### Voxel Downsampling Filter
The first filter applied to the pcl image is a voxel filter. As mentioned in the course lessons, *"RGB-D cameras provide feature rich and particularly dense point clouds, meaning, more points are packed in per unit volume than, for example, a Lidar point cloud. Running computation on a full resolution point cloud can be slow and may not yield any improvement on results obtained using a more sparsely sampled point cloud."* To reduce the point cloud density, and the computing time by extension, a voxel grid downsampling filter is applied. The filter essentially breaks down the point cloud into smaller cubes. Within each cube, the filter takes the spatial average of all the points. The average becomes an output point, the voxel. Now the original pcl image is comprised of multiple voxels, or volumetric points, reducing the amount of points representing the data. The dimensions of the voxel cube can be increased or decreased accordingly. The smaller the voxel is, the more accurate is the representation of the image, but the longer is the computation time. So it's a trade off of sorts. Ideally, you want to choose the voxel dimensions so that the image is well represented without much loss in the data, while maintaining a relatively short computing time. 

```py
vox = cloud.make_voxel_grid_filter()
LEAF_SIZE = 0.005
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
cloud_filtered = vox.filter()
```

As shown in the code block above the voxel dimension used, `LEAF_SIZE` is chosen to be 0.005m. This value has resulted in a decent output under a very reasonable computing time.

#### Passthrough Filter
The next filter applied is a pass through filter. This filter basically crops the pcl in 3d space. You can choose in which axis to crop and then from where to where along the axis. The point of doing this is to eliminate unnecessary data from being processed further on. For instance, the image below shows what the robot initially sees. It is clear that the robot needs only consider the table itself and the objects above it. It has no need to look at the table's leg, which could result in misidentification problems later on. So the pcl is essentially cropped in the z-axis to result in the second image.

![TableAndObjectsUncropped](./WriteupImages/TableAndObjectsUncropped)

![TableAndObjectsCropped](https://github.com/AGKhalil/RoboND-Perception-Project/blob/master/WriteupImages/TableAndObjectsCropped.png)

The code block below shows the filter in action as well as the cropping value chosen for the project. In the project, the robot kept seeing objects on the left and right. It would then go on to cluster those points. To avoid that from occurring, an additional passthrough filter was added, in the y-direction.

```py
passthrough_z = cloud_filtered.make_passthrough_filter()
passthrough_z.set_filter_field_name('z')
axis_min_z = 0.6
axis_max_z = 0.95
passthrough_z.set_filter_limits(axis_min_z, axis_max_z)
cloud_filtered = passthrough_z.filter()
    
passthrough_y = cloud_filtered.make_passthrough_filter()
passthrough_y.set_filter_field_name('y')
axis_min_y = -0.5
axis_max_y = 0.5
passthrough_y.set_filter_limits(axis_min_y, axis_max_y)
cloud_filtered = passthrough_y.filter()
```
#### RANSAC Filter
A Random Sample Consensus, or RANSAC, filter is great at identifying the points in a set of data that belong to a certain geometric shape. It identifies the inliers and outliers of the shape in question. The robot does not need to look at the table to recognize the objects, in fact, it may make it harder for the clustering process. Therefore, the table is removed using a RANSAC filter. We know that the table must fit in a plane shape, so the RANSAC filter is applied to the pcl with that in mind. All the points comprising the table plane are inliers and any other point is an outlier; thus, effectively separating both entities from one another. Now that we know what points make up the table, we can just remove them.

The code block below shows the RANSAC filter applied to the project. It is clear that the maximum distance between each point in the plane criteria was set to 0.04m. This value was effective in identifying the table points. Then the table points are extracted from the pcl.

```py
seg = cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
max_distance = 0.04
seg.set_distance_threshold(max_distance)
inliers, coefficients = seg.segment()
extracted_table = cloud_filtered.extract(inliers, negative=False)
extracted_objects = cloud_filtered.extract(inliers, negative=True)
```
#### Outlier Removal Filter
All this filtering is great, but it does nothing to eliminate noise from the pcl. That's where this filter comes into play. This filter is pretty simple, it takes a point in the pcl and it looks at a certain number of points surrounding it. Any point that is a certain distance away is considered an outlier. This filter really cleans up the data for further analysis. The image below shows a simple example of how this filter works.

![OutlierRemovalFilter](https://github.com/AGKhalil/RoboND-Perception-Project/blob/master/WriteupImages/OutlierRemovalFilter.png)

In the project the following outlier removal filter was applied. The filter is set to check 100 neighboring points to each point under inspection. Any point a mean distance away larger than the mean distance + x * standard deviation is an outlier. x was chosen as 0.5, which results in a very could filter.

```py
outlier_filter = cloud.make_statistical_outlier_filter()
outlier_filter.set_mean_k(100)
x = 0.5
outlier_filter.set_std_dev_mul_thresh(x)
cloud = outlier_filter.filter()
```

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

After all the filters from exercise 1 are applied, we are left with a pcl comprised of a set of objects. Yet, the robot does not yet know how many objects are out there, or which points belong to a single object. Therefore, a Density-Based Spatial Clustering of Applications with Noise algorithm, DBSCAN, is used. This algorithm looks at a set of data and calculates the Euclidean distance between a point and its neighbors. If that distance is below a certain specified value, the point and its neighbor belong in the same cluster. Furthermore, the minimum and maximum sizes of the cluster are specified.

In the project a clustering distance of 0.01m was used. It is a small value ensuring that no points from neighboring objects would be contained in the same cluster. Moreover, the minimum and maximum sizes of the cluster are chosen as 10 and 2500. I don't really think these two values are of any significant importance. Since the pcl is filtered from any noise, there are no random 5 points near each other that can be mistaken for a cluster. So as long as the minimum number is below that of the smallest object, there is no problem. Same thing for the max number, as long as it is larger than the largest object, there is no problem.

After the clustering, each cluster is corresponded to a random color to aid in the visual representation.

```py
# Euclidean Clustering
white_cloud = XYZRGB_to_XYZ(extracted_objects)
tree = white_cloud.make_kdtree()
ec = white_cloud.make_EuclideanClusterExtraction()
# Set tolerances for distance threshold
# as well as minimum ans maximum cluster size (in points)
ec.set_ClusterTolerance(0.01)
ec.set_MinClusterSize(10)
ec.set_MaxClusterSize(2500)
# Search the k-d tree for clusters
ec.set_SearchMethod(tree)
# Extract indices for each of the discovered clusters
cluster_indices = ec.Extract()

# Create Cluster-Mask Point Cloud to visualize each cluster separately
# Assign a color corresponding to each segmented object in scene
cluster_color = get_color_list(len(cluster_indices))
color_cluster_point_list = []
for j, indices in enumerate(cluster_indices):
  for i, indice in enumerate(indices):
    color_cluster_point_list.append([white_cloud[indice][0],
                                     white_cloud[indice][1],
                                     white_cloud[indice][2],
                                     rgb_to_float(cluster_color[j])])
```

#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Now that we have each object represented by a single cluster respectively, we can go about object recognition. A machine learning algorithm called Support Vector Machine, SVM, is used. For the purpose of this project the algorithm works in this way:

- For each object a set of random orientations is created, the training set, along with the corresponding labels for each object
- For each orientation the algorithm analyzes what it sees based on the color composition and normal vector distribution, these represent the features of each object
- A histogram is created for each object representing the composition of each feature
- The histograms are normalized to make sure that all features are compared on the same scale
- Now if the algorithm is faced with a random object, it will compare what it sees against its training data and classify the object accordingly

For a deeper insight please refer to [`capture_feature.py`](https://github.com/AGKhalil/RoboND-Perception-Project/blob/master/sensor_stick/scripts/capture_features.py) and to [`features.py`](https://github.com/AGKhalil/RoboND-Perception-Project/blob/master/sensor_stick/src/sensor_stick/features.py).

The screenshot below shows the confusion matrix for one of the worlds, `test1.world.`. As shown, there are two matrices, the first, Figure 1, shows how many times each item has been labeled correctly as predicted. The second, Figure 2, shows the same data but normalized. For instance, Figure 1 shows that biscuits have been recognized 196 times out of 200 attempts, and Figure 2 shows what that is as a proportion, 0.98. The higher is the success percentage, the darker will the matrix elements be. So we want a very nice and linear relationship between the predicted and actual labels. These two figures show a very high success rate. So the chances that the algorithm will succeed in simulation is good.

![ConfusionMatrices](https://github.com/AGKhalil/RoboND-Perception-Project/blob/master/WriteupImages/ConfusionMatrices.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

Here all the filtering, clustering, and object recognition learned in exercises 1-3 are applied. Three test worlds are given and the robot is supposed to recognize the objects before it so it can decide where to place them. 

The main difference between this script and the excursus 1-3 scripts is that it also outputs the results in a `.yaml` file.

This has been a very educational project and I really enjoyed going through it. I did learn that machine learning is a really cool thing, but it needs a lot of training. I do hope to pursue the additional challenge of this project in the future. If I am to keep going with this project however, I would optimize my training technique more to reduce the computing time. Right now It takes about 40mins to train each set and that is long. I am using hsv color recognition and 200 iterations. 

All in all, this has been fun!