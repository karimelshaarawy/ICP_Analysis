#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"

/*
 * ICP Optimizer Configuration:
 * 
 * Choose one of the following ICP implementations:
 * - USE_LINEAR_ICP = 1: Uses LinearICPOptimizer (fast, works well for small rotations)
 * - USE_LM_ICP = 1: Uses LevenbergMarquardtICPOptimizer (robust, handles large rotations)
 * - Both = 0: Uses CeresICPOptimizer (default, most robust but requires Ceres)
 * 
 * Constraint Type:
 * - USE_POINT_TO_PLANE = 1: Uses point-to-plane constraints (usually better for surfaces)
 * - USE_POINT_TO_PLANE = 0: Uses point-to-point constraints (faster, works well for point clouds)
 * 
 * Examples:
 * - For bunny alignment with LM-ICP and point-to-plane: USE_LM_ICP=1, USE_POINT_TO_PLANE=1
 * - For room reconstruction with linear ICP: USE_LINEAR_ICP=1, USE_POINT_TO_PLANE=0
 */

#define SHOW_BUNNY_CORRESPONDENCES 0

#define USE_POINT_TO_PLANE  0
#define USE_LINEAR_ICP      0
#define USE_LM_ICP          0
#define USE_SYMMETRIC_ICP   1 // Set to 1 to use Symmetric ICP


#define RUN_SHAPE_ICP		0
#define RUN_SEQUENCE_ICP	1

void debugCorrespondenceMatching() {
	// Load the source and target mesh.
	const std::string filenameSource = std::string("../../Data/bunny_part2_trans.off");
	const std::string filenameTarget = std::string("../../Data/bunny_part1.off");

	SimpleMesh sourceMesh;
	if (!sourceMesh.loadMesh(filenameSource)) {
		std::cout << "Mesh file wasn't read successfully." << std::endl;
		return;
	}

	SimpleMesh targetMesh;
	if (!targetMesh.loadMesh(filenameTarget)) {
		std::cout << "Mesh file wasn't read successfully." << std::endl;
		return;
	}

	PointCloud source{ sourceMesh };
	PointCloud target{ targetMesh };
	
	// Search for matches using FLANN.
	std::unique_ptr<NearestNeighborSearch> nearestNeighborSearch = std::make_unique<NearestNeighborSearchFlann>();
	nearestNeighborSearch->setMatchingMaxDistance(0.0001f);
	nearestNeighborSearch->buildIndex(target.getPoints());
	auto matches = nearestNeighborSearch->queryMatches(source.getPoints());

	// Visualize the correspondences with lines.
	SimpleMesh resultingMesh = SimpleMesh::joinMeshes(sourceMesh, targetMesh, Matrix4f::Identity());
	auto sourcePoints = source.getPoints();
	auto targetPoints = target.getPoints();

	for (unsigned i = 0; i < 100; ++i) { // sourcePoints.size()
		const auto match = matches[i];
		if (match.idx >= 0) {
			const auto& sourcePoint = sourcePoints[i];
			const auto& targetPoint = targetPoints[match.idx];
			resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::cylinder(sourcePoint, targetPoint, 0.002f, 2, 15), resultingMesh, Matrix4f::Identity());
		}
	}

	resultingMesh.writeMesh(std::string("correspondences.off"));
}

int alignBunnyWithICP() {
	// Load the source and target mesh.
	const std::string filenameSource = std::string("/config/workspace/ICP_Analysis /Data/Data/bunny_part2_trans.off");
	const std::string filenameTarget = std::string("/config/workspace/ICP_Analysis /Data/Data/bunny_part1.off");

	SimpleMesh sourceMesh;
	if (!sourceMesh.loadMesh(filenameSource)) {
		std::cout << "Mesh file wasn't read successfully at location: " << filenameSource << std::endl;
		return -1;
	}

	SimpleMesh targetMesh;
	if (!targetMesh.loadMesh(filenameTarget)) {
		std::cout << "Mesh file wasn't read successfully at location: " << filenameTarget << std::endl;
		return -1;
	}

	// Estimate the pose from source to target mesh with ICP optimization.
	ICPOptimizer* optimizer = nullptr;
	if (USE_LINEAR_ICP) {
		std::cout << "Using Linear ICP Optimizer" << std::endl;
		optimizer = new LinearICPOptimizer();
	}
	else if (USE_LM_ICP) {
		std::cout << "Using Levenberg-Marquardt ICP Optimizer" << std::endl;
		optimizer = new LevenbergMarquardtICPOptimizer();
		// Configure LM-specific parameters
		LevenbergMarquardtICPOptimizer* lmOptimizer = static_cast<LevenbergMarquardtICPOptimizer*>(optimizer);      // Fewer iterations per ICP step  // More lenient threshold         // Fewer ICP iterations
	}
	else {
		std::cout << "Using Ceres ICP Optimizer" << std::endl;
		optimizer = new AlternatingSymmetricICPOptimizer();
	}
	
	optimizer->setMatchingMaxDistance(0.0003f);
	if (USE_POINT_TO_PLANE) {
		optimizer->usePointToPlaneConstraints(true);
		optimizer->setNbOfIterations(10);
	}
	else {
		optimizer->usePointToPlaneConstraints(false);
		optimizer->setNbOfIterations(10);
	}

	PointCloud source{ sourceMesh };
	PointCloud target{ targetMesh };

    Matrix4f estimatedPose = Matrix4f::Identity();
	optimizer->estimatePose(source, target, estimatedPose);
	
	// Visualize the resulting joined mesh. We add triangulated spheres for point matches.
	SimpleMesh resultingMesh = SimpleMesh::joinMeshes(sourceMesh, targetMesh, estimatedPose);
	if (SHOW_BUNNY_CORRESPONDENCES) {
		for (const auto& sourcePoint : source.getPoints()) {
			resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::sphere(sourcePoint, 0.001f), resultingMesh, estimatedPose);
		}
		for (const auto& targetPoint : target.getPoints()) {
			resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::sphere(targetPoint, 0.001f, Vector4uc(255, 255, 255, 255)), resultingMesh, Matrix4f::Identity());
		}
	}
	resultingMesh.writeMesh(std::string("bunny_icp.off"));
	std::cout << "Resulting mesh written." << std::endl;

	delete optimizer;

	return 0;
}

int reconstructRoom() {
		std::cout << "lolo..." << std::endl;

	std::string filenameIn = std::string("../../../Data/rgbd_dataset_freiburg1_xyz/");
	std::string filenameBaseOut = std::string("mesh_");

	// Load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.init(filenameIn)) {
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// We store a first frame as a reference frame. All next frames are tracked relatively to the first frame.
	sensor.processNextFrame();
	PointCloud target{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(),4};

	// Setup the optimizer.
	ICPOptimizer* optimizer = nullptr;
	#if USE_SYMMETRIC_ICP
    std::cout << "Using Symmetric ICP Optimizer" << std::endl;
    optimizer = new AlternatingSymmetricICPOptimizer();
#elif USE_LINEAR_ICP
    std::cout << "Using Linear ICP Optimizer" << std::endl;
    optimizer = new LinearICPOptimizer();
#elif USE_LM_ICP
    std::cout << "Using Levenberg-Marquardt ICP Optimizer" << std::endl;
    optimizer = new LevenbergMarquardtICPOptimizer();
    // Configure LM-specific parameters if needed (uncomment and adjust)
    // LevenbergMarquardtICPOptimizer* lmOptimizer = static_cast<LevenbergMarquardtICPOptimizer*>(optimizer);
    // lmOptimizer->setNbOfIterations(5); // Fewer iterations per ICP step
    // lmOptimizer->setLambda(0.01f);      // More lenient threshold
    // lmOptimizer->setNbOfICPIterations(5); // Fewer ICP iterations
#else
    std::cout << "Using Ceres ICP Optimizer" << std::endl;
    optimizer = new CeresICPOptimizer();
#endif

    // Set common optimizer parameters
    optimizer->setMatchingMaxDistance(0.0003f);
    if (USE_POINT_TO_PLANE) {
        optimizer->usePointToPlaneConstraints(true);
        optimizer->setNbOfIterations(15);
    }
    else {
        optimizer->usePointToPlaneConstraints(false);
        optimizer->setNbOfIterations(15);
    }
	// We store the estimated camera poses.
	std::vector<Matrix4f> estimatedPoses;
	Matrix4f currentCameraToWorld = Matrix4f::Identity();
	estimatedPoses.push_back(currentCameraToWorld.inverse());

	int i = 0;
	const int iMax = 22;
	while (sensor.processNextFrame() && i <= iMax) {
		float* depthMap = sensor.getDepth();
		Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();
		Matrix4f depthExtrinsics = sensor.getDepthExtrinsics();

		// Estimate the current camera pose from source to target mesh with ICP optimization.
		// We downsample the source image to speed up the correspondence matching.
		PointCloud source{ sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getDepthExtrinsics(), sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), 4 };
		optimizer->estimatePose(source, target, currentCameraToWorld);
		
		// Invert the transformation matrix to get the current camera pose.
		Matrix4f currentCameraPose = currentCameraToWorld.inverse();
		std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
		estimatedPoses.push_back(currentCameraPose);

		if (i % 5 == 0) {
			// We write out the mesh to file for debugging.
			SimpleMesh currentDepthMesh{ sensor, currentCameraPose, 0.1f };
			SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
			SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

			std::stringstream ss;
			ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
			std::cout << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off" << std::endl;
			if (!resultingMesh.writeMesh(ss.str())) {
				std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
				return -1;
			}
		}
		
		i++;
	}

	delete optimizer;

	return 0;
}

int main() {
	int result = 0;
	if (RUN_SHAPE_ICP)
		result += alignBunnyWithICP();
	if (RUN_SEQUENCE_ICP)
		result += reconstructRoom();

	return result;
}
