# Aerial Object Detection
Creating an algorithm to classify objects from a top down aerial view supplied with google maps or a drone live feed.

## Purpose
Assist companies in estimation and tracking for different occasions such as:
- Roof size estimation (Completed)
- Lawn size coverage (Development)
- Shortest path between two points after classifying objects that reduce mobility. (Not Started)

## Objectives
- Vehicle(s)
  - Gather sampling data from ArcGS for model training.
  - Construct a basic algorithm with K-means Clustering.
  
## Progress
__Data preprocessing:__
Detect boundary contour
Crop inside of boundary

__Modeling:__
Generated a model to classify houses and create a contour surrounding the house for roof measurement and property distribution.

__Measurements:__
Roof Area ~ (Detected Contour Area / Original Contour Area) * Property Area
