#pragma once

// The Google logging library (GLOG), used in Ceres, has a conflict with Windows defined constants. This definitions prevents GLOG to use the same constants
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <flann/flann.hpp>

#include "SimpleMesh.h"
#include "NearestNeighbor.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"

#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Type definitions for 6D pose representation
typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 3, 6> Matrix3x6f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;

using Vector3d = Eigen::Vector3d;
using Matrix4d = Eigen::Matrix4d;
using Matrix3d = Eigen::Matrix3d;

/**
 * Helper methods for writing Ceres cost functions.
 */
template <typename T>
static inline void fillVector(const Vector3f& input, T* output) {
    output[0] = T(input[0]);
    output[1] = T(input[1]);
    output[2] = T(input[2]);
}


/**
 * Pose increment is only an interface to the underlying array (in constructor, no copy
 * of the input array is made).
 * Important: Input array needs to have a size of at least 6.
 */
template <typename T>
class PoseIncrement {
public:
    explicit PoseIncrement(T* const array) : m_array{ array } { }

    void setZero() {
        for (int i = 0; i < 6; ++i)
            m_array[i] = T(0);
    }

    T* getData() const {
        return m_array;
    }

        void getNormal(const T* normal, T* rotatedNormal) const {
    ceres::AngleAxisRotatePoint(m_array, normal, rotatedNormal);
    }



    /**
     * Applies the pose increment onto the input point and produces transformed output point.
     * Important: The memory for both 3D points (input and output) needs to be reserved (i.e. on the stack)
     * beforehand).
     */
    void apply(T* inputPoint, T* outputPoint) const {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        const T* rotation = m_array;
        const T* translation = m_array + 3;

        T temp[3];
        ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

        outputPoint[0] = temp[0] + translation[0];
        outputPoint[1] = temp[1] + translation[1];
        outputPoint[2] = temp[2] + translation[2];
    }
    void applyInverse(T* inputPoint, T* outputPoint) const {
    // pose[0,1,2] is angle-axis rotation.
    // pose[3,4,5] is translation.
    const T* rotation = m_array;
    const T* translation = m_array + 3;

    // First, subtract the translation
    T temp[3];
    temp[0] = inputPoint[0] - translation[0];
    temp[1] = inputPoint[1] - translation[1];
    temp[2] = inputPoint[2] - translation[2];

    // Then apply the inverse rotation (negative angle-axis)
    T inverse_rotation[3];
    inverse_rotation[0] = -rotation[0];
    inverse_rotation[1] = -rotation[1];
    inverse_rotation[2] = -rotation[2];

    ceres::AngleAxisRotatePoint(inverse_rotation, temp, outputPoint);
}

    /**
     * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
     * transformation 4x4 matrix.
     */
    static Matrix4f convertToMatrix(const PoseIncrement<double>& poseIncrement) {
        // pose[0,1,2] is angle-axis rotation.
        // pose[3,4,5] is translation.
        double* pose = poseIncrement.getData();
        double* rotation = pose;
        double* translation = pose + 3;

        // Convert the rotation from SO3 to matrix notation (with column-major storage).
        double rotationMatrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

        // Create the 4x4 transformation matrix.
        Matrix4f matrix;
        matrix.setIdentity();
        matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
        matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
        matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);

        return matrix;
    }

private:
    T* m_array;
};

/**
 * Optimization constraints.
 */
class PointToPointConstraint {
public:
    PointToPointConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, const float weight) :
        m_sourcePoint{ sourcePoint },
        m_targetPoint{ targetPoint },
        m_weight{ weight }
    { }

    template<typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Implemented the point-to-point cost function.
        // The resulting 3D residual should be stored in residuals array. To apply the pose
        // increment (pose parameters) to the source point, you can use the PoseIncrement
        // class.
        // Important: Ceres automatically squares the cost function.
        T sourcePoint[3];
        fillVector(m_sourcePoint, sourcePoint);

        PoseIncrement<T> poseIncrement =PoseIncrement<T>(const_cast< T* const>(pose));
        T sourcePointTransformed[3];
        poseIncrement.apply(sourcePoint, sourcePointTransformed);

        // Calculate the residual: (R*s + t) - d
        // Where R*s + t is sourcePointTransformed, d is m_targetPoint
        residuals[0] = T(m_weight) * (sourcePointTransformed[0] - T(m_targetPoint[0]));
        residuals[1] = T(m_weight) * (sourcePointTransformed[1] - T(m_targetPoint[1]));
        residuals[2] = T(m_weight) * (sourcePointTransformed[2] - T(m_targetPoint[2]));

        return true;
    }

    static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, const float weight) {
        return new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
            new PointToPointConstraint(sourcePoint, targetPoint, weight)
            );
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const float m_weight;
    const float LAMBDA = 0.1f;
};

class PointToPlaneConstraint {
public:
    PointToPlaneConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, const Vector3f& targetNormal, const float weight) :
        m_sourcePoint{ sourcePoint },
        m_targetPoint{ targetPoint },
        m_targetNormal{ targetNormal },
        m_weight{ weight }
    { }
template <typename T>
bool operator()(const T* const pose, T* residuals) const {
    T source_point[3];
    fillVector(m_sourcePoint,source_point);
    PoseIncrement<T> poseIncrement =PoseIncrement<T>(const_cast< T* const>(pose));
    T sourcePointTransformed[3];
    poseIncrement.apply(source_point,sourcePointTransformed);

    residuals[0]= T(LAMBDA) * T(m_weight) * (
        (sourcePointTransformed[0] - T(m_targetPoint[0])) * T(m_targetNormal[0]) +
        (sourcePointTransformed[1] - T(m_targetPoint[1])) * T(m_targetNormal[1]) +
        (sourcePointTransformed[2] - T(m_targetPoint[2])) * T(m_targetNormal[2])

    );
    return true;

}

    static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, const Vector3f& targetNormal, const float weight) {
        return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
            new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, weight)
            );
    }

protected:

template <typename T>
void applyPoseTransformation(const T* pose_matrix, const T* point, T* result) const {
    // Apply 4x4 transformation matrix to 3D point
    result[0] = pose_matrix[0] * point[0] + pose_matrix[1] * point[1] + 
               pose_matrix[2] * point[2] + pose_matrix[3];
    result[1] = pose_matrix[4] * point[0] + pose_matrix[5] * point[1] + 
               pose_matrix[6] * point[2] + pose_matrix[7];
    result[2] = pose_matrix[8] * point[0] + pose_matrix[9] * point[1] + 
               pose_matrix[10] * point[2] + pose_matrix[11];
}
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const Vector3f m_targetNormal;
    const float m_weight;
    const float LAMBDA = 1.0f;
};

class SymmetricPointToPlaneConstraint {
public:
    SymmetricPointToPlaneConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, 
                                   const Vector3f& sourceNormal, const Vector3f& targetNormal, 
                                   const float weight) :
        m_sourcePoint{ sourcePoint },
        m_targetPoint{ targetPoint },
        m_sourceNormal{ sourceNormal },
        m_targetNormal{ targetNormal },
        m_weight{ weight }
    { }

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        T source_point[3];
        fillVector(m_sourcePoint, source_point);
        
        T target_point[3];
        fillVector(m_targetPoint, target_point);
        
        PoseIncrement<T> poseIncrement = PoseIncrement<T>(const_cast<T* const>(pose));
        
        T sourcePointTransformed[3];
        poseIncrement.apply(source_point, sourcePointTransformed);
        
        T targetPointTransformed[3];
        poseIncrement.applyInverse(target_point, targetPointTransformed);
        
        // First constraint: transformed source point to target plane
        T residual1 = (sourcePointTransformed[0] - T(m_targetPoint[0])) * T(m_targetNormal[0]) +
                      (sourcePointTransformed[1] - T(m_targetPoint[1])) * T(m_targetNormal[1]) +
                      (sourcePointTransformed[2] - T(m_targetPoint[2])) * T(m_targetNormal[2]);
        
        // Second constraint: inverse-transformed target point to source plane  
        T residual2 = (targetPointTransformed[0] - T(m_sourcePoint[0])) * T(m_sourceNormal[0]) +
                      (targetPointTransformed[1] - T(m_sourcePoint[1])) * T(m_sourceNormal[1]) +
                      (targetPointTransformed[2] - T(m_sourcePoint[2])) * T(m_sourceNormal[2]);
        
        residuals[0] = T(LAMBDA) * T(m_weight) * residual1;
        residuals[1] = T(LAMBDA) * T(m_weight) * residual2;
        
        return true;
    }

    static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, 
                                      const Vector3f& sourceNormal, const Vector3f& targetNormal, 
                                      const float weight) {
        return new ceres::AutoDiffCostFunction<SymmetricPointToPlaneConstraint, 2, 6>(
            new SymmetricPointToPlaneConstraint(sourcePoint, targetPoint, sourceNormal, targetNormal, weight)
        );
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const Vector3f m_sourceNormal;
    const Vector3f m_targetNormal;
    const float m_weight;
    const float LAMBDA = 1.0f;
};

   
/**
 * ICP optimizer - Abstract Base Class
 */
class ICPOptimizer {
public:
    ICPOptimizer() :
        m_bUsePointToPlaneConstraints{ true },
        m_nIterations{ 20 },
        m_nearestNeighborSearch{ std::make_unique<NearestNeighborSearchFlann>() }
    { }

    void setMatchingMaxDistance(float maxDistance) {
        m_nearestNeighborSearch->setMatchingMaxDistance(maxDistance);
    }

    void usePointToPlaneConstraints(bool bUsePointToPlaneConstraints) {
        m_bUsePointToPlaneConstraints = bUsePointToPlaneConstraints;
    }

    void setNbOfIterations(unsigned nIterations) {
        m_nIterations = nIterations;
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) = 0;

protected:
    bool m_bUsePointToPlaneConstraints;
    unsigned m_nIterations;
    std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

    std::vector<Vector3f> transformPoints(const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose) {
        std::vector<Vector3f> transformedPoints;
        transformedPoints.reserve(sourcePoints.size());

        const auto rotation = pose.block(0, 0, 3, 3);
        const auto translation = pose.block(0, 3, 3, 1);

        for (const auto& point : sourcePoints) {
            transformedPoints.push_back(rotation * point + translation);
        }

        return transformedPoints;
    }

    std::vector<Vector3f> transformNormals(const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose) {
        std::vector<Vector3f> transformedNormals;
        transformedNormals.reserve(sourceNormals.size());

        const auto rotation = pose.block(0, 0, 3, 3);

        for (const auto& normal : sourceNormals) {
            transformedNormals.push_back(rotation.inverse().transpose() * normal);
        }

        return transformedNormals;
    }

    void pruneCorrespondences(const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches) {
        const unsigned nPoints = sourceNormals.size();

        for (unsigned i = 0; i < nPoints; i++) {
            Match& match = matches[i];
            if (match.idx >= 0) {
                const auto& sourceNormal = sourceNormals[i];
                const auto& targetNormal = targetNormals[match.idx];

                // TODO: Invalidate the match (set it to -1) if the angle between the normals is greater than 60
                if(sourceNormal.dot(targetNormal)< 0.5)
                match.idx= -1;
                
            }
        }
    }
};


/**
 * ICP optimizer - using Ceres for optimization.
 */
class CeresICPOptimizer : public ICPOptimizer {
public:
    CeresICPOptimizer() {}

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) override {
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        m_nearestNeighborSearch->buildIndex(target.getPoints());

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        // We optimize on the transformation in SE3 notation: 3 parameters for the axis-angle vector of the rotation (its length presents
        // the rotation angle) and 3 parameters for the translation vector. 
        double incrementArray[6];
        auto poseIncrement = PoseIncrement<double>(incrementArray);
        poseIncrement.setZero();

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            clock_t begin = clock();

            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

            auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            pruneCorrespondences(transformedNormals, target.getNormals(), matches);

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

            // Prepare point-to-point and point-to-plane constraints.
            ceres::Problem problem;
            prepareConstraints(transformedPoints, target.getPoints(), target.getNormals(), matches, poseIncrement, problem);

            // Configure options for the solver.
            ceres::Solver::Options options;
            configureSolver(options);

            // Run the solver (for one iteration).
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;
            //std::cout << summary.FullReport() << std::endl;

            // Update the current pose estimate (we always update the pose from the left, using left-increment notation).
            Matrix4f matrix = PoseIncrement<double>::convertToMatrix(poseIncrement);
            estimatedPose = PoseIncrement<double>::convertToMatrix(poseIncrement) * estimatedPose;
            poseIncrement.setZero();

            std::cout << "Optimization iteration done." << std::endl;
        }

        // Store result
        initialPose = estimatedPose;
    }


private:
    void configureSolver(ceres::Solver::Options& options) {
        // Ceres options.
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = 1;
        options.max_num_iterations = 1;
        options.num_threads = 8;
    }

    void prepareConstraints(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals, const std::vector<Match> matches, const PoseIncrement<double>& poseIncrement, ceres::Problem& problem) const {
        const unsigned nPoints = sourcePoints.size();

        for (unsigned i = 0; i < nPoints; ++i) {
            const auto match = matches[i];
            if (match.idx >= 0) {
                const auto& sourcePoint = sourcePoints[i];
                const auto& targetPoint = targetPoints[match.idx];

                if (!sourcePoint.allFinite() || !targetPoint.allFinite())
                    continue;

                // TODO: Create a new point-to-point cost function and add it as constraint (i.e. residual block) 
                // to the Ceres problem.

                ceres::CostFunction* pointToPointCost = PointToPointConstraint::create(sourcePoint, targetPoint,1.0f);
            problem.AddResidualBlock(pointToPointCost, nullptr, poseIncrement.getData());


                if (m_bUsePointToPlaneConstraints) {
                    const auto& targetNormal = targetNormals[match.idx];

                    if (!targetNormal.allFinite())
                        continue;

                    // TODO: Create a new point-to-plane cost function and add it as constraint (i.e. residual block) 
                    // to the Ceres problem.
                    ceres::CostFunction* pointToPlaneCost = PointToPlaneConstraint::create(sourcePoint, targetPoint, targetNormal,1.0f);
                problem.AddResidualBlock(pointToPlaneCost, nullptr, poseIncrement.getData());


                }
            }
        }
    }
};

/**
 * ICP optimizer - using linear least-squares for optimization.
 */
 
class SymmetricICPOptimizer : public ICPOptimizer {
public:
    /**
     * @brief Constructor for SymmetricICPOptimizer.
     * The base class ICPOptimizer's constructor will initialize m_nearestNeighborSearch.
     */
    SymmetricICPOptimizer() {}

    /**
     * @brief Destructor for SymmetricICPOptimizer.
     * Default destructor is sufficient as m_nearestNeighborSearch is a unique_ptr
     * managed by the base class.
     */
    ~SymmetricICPOptimizer() = default;

    /**
     * @brief Estimates the rigid body transformation (rotation and translation)
     * from the source point cloud to the target point cloud using
     * alternating symmetric linear ICP.
     *
     * @param source The source point cloud.
     * @param target The target point cloud.
     * @param transformation The initial guess for the transformation matrix (updated in place).
     */
    void estimatePose(const PointCloud& source, const PointCloud& target, Eigen::Matrix4f& transformation) override {
        // The initial estimate can be given as an argument.
        Eigen::Matrix4f estimatedPose = transformation;

        for (int i = 0; i < m_nIterations; ++i) {
            std::cout << "Symmetric ICP Iteration " << i + 1 << "/" << m_nIterations << std::endl;
            clock_t begin = clock();

            // 1. Transform source cloud to current estimated pose
            auto transformedSourcePoints = transformPoints(source.getPoints(), estimatedPose);
            auto transformedSourceNormals = transformNormals(source.getNormals(), estimatedPose);

            // Vectors to store combined correspondences for the current iteration
            std::vector<Eigen::Vector3f> combinedSourcePoints;
            std::vector<Eigen::Vector3f> combinedSourceNormals; // NEW: To store normals corresponding to combinedSourcePoints
            std::vector<Eigen::Vector3f> combinedTargetPoints;
            std::vector<Eigen::Vector3f> combinedTargetNormals;

            // --- A. Source-to-Target (S->T) Correspondences ---
            std::cout << "  Matching S->T ..." << std::endl;
            m_nearestNeighborSearch->buildIndex(target.getPoints()); // Build index on the static target cloud
            auto matches_ST = m_nearestNeighborSearch->queryMatches(transformedSourcePoints);
            // Prune using the normals of the query points (transformed source) and their matches (target)
            pruneCorrespondences(transformedSourceNormals, target.getNormals(), matches_ST);

            for (size_t j = 0; j < transformedSourcePoints.size(); ++j) {
                const auto& match = matches_ST[j];
                if (match.idx >= 0) {
                    combinedSourcePoints.push_back(transformedSourcePoints[j]);
                    combinedSourceNormals.push_back(transformedSourceNormals[j]); // Normal of transformed source point
                    combinedTargetPoints.push_back(target.getPoints()[match.idx]);
                    combinedTargetNormals.push_back(target.getNormals()[match.idx]); // Normal of target point
                }
            }

            // --- B. Target-to-Source (T->S) Correspondences ---
            std::cout << "  Matching T->S ..." << std::endl;
            m_nearestNeighborSearch->buildIndex(transformedSourcePoints); // Build index on the transformed source cloud
            auto matches_TS = m_nearestNeighborSearch->queryMatches(target.getPoints()); // Query from original target points
            // Prune using the normals of the query points (original target) and their matches (transformed source)
            pruneCorrespondences(target.getNormals(), transformedSourceNormals, matches_TS);

            for (size_t j = 0; j < target.getPoints().size(); ++j) {
                const auto& match = matches_TS[j];
                if (match.idx >= 0) {
                    // For T->S, the "source" point is from the original target cloud
                    combinedSourcePoints.push_back(target.getPoints()[j]);
                    combinedSourceNormals.push_back(target.getNormals()[j]); // Normal of original target point
                    // The "target" point is from the transformed source cloud
                    combinedTargetPoints.push_back(transformedSourcePoints[match.idx]);
                    combinedTargetNormals.push_back(transformedSourceNormals[match.idx]); // Normal of transformed source point
                }
            }

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "  Correspondence matching completed in " << elapsedSecs << " seconds." << std::endl;
            std::cout << "  Total combined correspondences: " << combinedSourcePoints.size() << std::endl;

            // Check if we have enough correspondences
            if (combinedSourcePoints.empty()) {
                std::cout << "  No correspondences found. Stopping ICP." << std::endl;
                break;
            }

            // Estimate the new pose (delta transformation) using Ceres for point-to-plane
            Eigen::Matrix4f deltaPose;
            if (m_bUsePointToPlaneConstraints) {
                // Call the new Ceres-based optimization function
                deltaPose = estimatePoseNonLinear(
                    combinedSourcePoints,
                    combinedTargetPoints,
                    combinedSourceNormals, // Pass source normals for the constraint
                    combinedTargetNormals
                );
            } else {
                deltaPose = estimatePosePointToPoint(combinedSourcePoints, combinedTargetPoints);
            }

            // Apply the delta transformation to the current estimated pose
            // deltaPose transforms points from the current estimated coordinate system.
            estimatedPose = deltaPose * estimatedPose;

            std::cout << "  Optimization iteration done. Current estimated pose: " << std::endl;
            std::cout << estimatedPose << std::endl;
        }

        transformation = estimatedPose; // Update the reference argument with the final pose
    }

private:
    Eigen::Matrix4f estimatePosePointToPoint(const std::vector<Eigen::Vector3f>& sourcePoints, const std::vector<Eigen::Vector3f>& targetPoints) {
        ProcrustesAligner procrustAligner;
        Eigen::Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);
        return estimatedPose;
    }

    // New function: Estimates the pose increment using Ceres Solver for symmetric point-to-plane
    Eigen::Matrix4f estimatePoseNonLinear(
        const std::vector<Eigen::Vector3f>& sourcePoints,
        const std::vector<Eigen::Vector3f>& targetPoints,
        const std::vector<Eigen::Vector3f>& sourceNormals, // Normals corresponding to sourcePoints
        const std::vector<Eigen::Vector3f>& targetNormals) {

        // The pose_increment will be estimated by Ceres. It represents the transformation
        // that moves points from the 'current' frame to the 'target' frame.
        // It's a 6-element array: [angle_axis_x, angle_axis_y, angle_axis_z, tx, ty, tz]
        double pose_increment[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Initialize with identity (no transformation)

        ceres::Problem problem;

        // Add residual blocks for each correspondence
        for (size_t i = 0; i < sourcePoints.size(); ++i) {
            if (i >= sourceNormals.size() || i >= targetNormals.size()) {
                // Should not happen if vectors are consistent, but good to check
                continue;
            }

            ceres::CostFunction* cost_function =
                SymmetricPointToPlaneConstraint::create(
                    sourcePoints[i],
                    targetPoints[i],
                    sourceNormals[i],
                    targetNormals[i],
                    1.0 // Weight. Could be dynamic based on distance, normal alignment, etc.
                );

            problem.AddResidualBlock(cost_function,
                                     new ceres::HuberLoss(0.1), // Robust loss function to handle outliers
                                     pose_increment);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR; // Or DENSE_SCHUR if you have structure
        options.minimizer_progress_to_stdout = true; // See optimization progress
        options.max_num_iterations = 50;            // Max inner iterations for Ceres
        options.function_tolerance = 1e-6;          // Stop if reduction in cost is small
        options.gradient_tolerance = 1e-10;
        options.num_threads = 4; // Use multiple threads if compiled with TBB/OpenMP

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // std::cout << summary.FullReport() << std::endl; // Uncomment for detailed Ceres report

        // Convert the estimated pose_increment (angle-axis + translation) back to Eigen::Matrix4f
        Eigen::Matrix4f deltaPose = Eigen::Matrix4f::Identity();

        // Convert Angle-Axis (Rodrigues) to Rotation Matrix
        Eigen::Vector3d angle_axis_vec(pose_increment[0], pose_increment[1], pose_increment[2]);
        Eigen::Matrix3d rotation_matrix_double;
        ceres::AngleAxisToRotationMatrix(angle_axis_vec.data(), rotation_matrix_double.data());

        deltaPose.block<3, 3>(0, 0) = rotation_matrix_double.cast<float>();
        deltaPose.block<3, 1>(0, 3) = Eigen::Vector3f(static_cast<float>(pose_increment[3]),
                                                     static_cast<float>(pose_increment[4]),
                                                     static_cast<float>(pose_increment[5]));

        return deltaPose;
    }
};

    
class LinearICPOptimizer : public ICPOptimizer {
public:
    LinearICPOptimizer() {}

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) override {
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        m_nearestNeighborSearch->buildIndex(target.getPoints());

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            clock_t begin = clock();

            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

            auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            pruneCorrespondences(transformedNormals, target.getNormals(), matches);

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

            std::vector<Vector3f> sourcePoints;
            std::vector<Vector3f> targetPoints;

            // Add all matches to the sourcePoints and targetPoints vector,
            // so that the sourcePoints[i] matches targetPoints[i]. For every source point,
            // the matches vector holds the index of the matching target point.
            for (int j = 0; j < transformedPoints.size(); j++) {
                const auto& match = matches[j];
                if (match.idx >= 0) {
                    sourcePoints.push_back(transformedPoints[j]);
                    targetPoints.push_back(target.getPoints()[match.idx]);
                }
            }

            // Estimate the new pose
            if (m_bUsePointToPlaneConstraints) {
                estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, target.getNormals()) * estimatedPose;
            }
            else {
                estimatedPose = estimatePosePointToPoint(sourcePoints, targetPoints) * estimatedPose;
            }

            std::cout << "Optimization iteration done." << std::endl;
        }

        // Store result
        initialPose = estimatedPose;
    }


private:
    Matrix4f estimatePosePointToPoint(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
        ProcrustesAligner procrustAligner;
        Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);

        return estimatedPose;
    }

    Matrix4f estimatePosePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals) {
    const unsigned nPoints = sourcePoints.size();

    // Build the system
    MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
    VectorXf b = VectorXf::Zero(4 * nPoints);

    for (unsigned i = 0; i < nPoints; ++i) {
         const auto& s = sourcePoints[i];
        const auto& d = targetPoints[i];
        const auto& n = targetNormals[i];

        // 1000: Add the point-to-plane constraints to the system
        A(4 * i, 0) = n.z() * s.y() - n.y() * s.z();
        A(4 * i, 1) = n.x() * s.z() - n.z() * s.x();
        A(4 * i, 2) = n.y() * s.x() - n.x() * s.y();
        A.block<1, 3>(4 * i, 3) = n;
        b(4 * i) = (d - s).dot(n);

        // TODO: Add the point-to-point constraints to the system
        A.block<3, 3>(4 * i + 1, 0) << 0.0f, s.z(), -s.y(),
                                      -s.z(), 0.0f, s.x(),
                                       s.y(), -s.x(), 0.0f;
        A.block<3, 3>(4 * i + 1, 3).setIdentity();
        b.segment<3>(4 * i + 1) = d - s;
    }

    

    // Apply a higher weight to point-to-plane correspondences
    float pointToPlaneWeight = 1.0f;
    float pointToPointWeight = 0.1f;
    A.block(0,0,4*nPoints,6) *= pointToPlaneWeight;
    b.segment(0,4*nPoints) *= pointToPlaneWeight;

    // TODO: Solve the system
    MatrixXf ATA = A.transpose() * A;
    VectorXf ATb = A.transpose() * b;

    JacobiSVD<MatrixXf> svd(ATA, ComputeFullU | ComputeFullV);
    VectorXf x = svd.solve(ATb);

    // Build the pose matrix
    float alpha = x(0), beta = x(1), gamma = x(2);
    Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix()
                      * AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix()
                      * AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();


    // TODO: Build the pose matrix. Your original code for this was also correct.
   Vector3f translation=x.tail(3);

   Matrix4f estimatedPose =Matrix4f::Identity();
   estimatedPose.block<3,3>(0,0) =rotation;
   estimatedPose.block<3,1>(0,3) =translation;



    return estimatedPose;
    }
};


/**
 * @struct PointToPointError
 * @brief Ceres cost functor for point-to-point ICP error.
 * The residual is the 3D vector difference between the transformed source and the target point.
 */
struct PointToPointError {
    const Vector3d source_point;
    const Vector3d target_point;

    PointToPointError(const Vector3d& source, const Vector3d& target)
        : source_point(source), target_point(target) {}

    template <typename T>
    bool operator()(const T* const pose_increment, T* residuals) const {
        // The pose_increment is a 6-element array:
        // - First 3 elements are an angle-axis rotation vector.
        // - Last 3 elements are a translation vector.
        
        T source_point_T[3] = {T(source_point.x()), T(source_point.y()), T(source_point.z())};
        T transformed_point_T[3];

        // Apply the rotation and translation increment.
        ceres::AngleAxisRotatePoint(pose_increment, source_point_T, transformed_point_T);
        transformed_point_T[0] += pose_increment[3];
        transformed_point_T[1] += pose_increment[4];
        transformed_point_T[2] += pose_increment[5];

        // The residual is the difference between the target and the transformed source.
        residuals[0] = T(target_point.x()) - transformed_point_T[0];
        residuals[1] = T(target_point.y()) - transformed_point_T[1];
        residuals[2] = T(target_point.z()) - transformed_point_T[2];

        return true;
    }
};

/**
 * @struct PointToPlaneError
 * @brief Ceres cost functor for point-to-plane ICP error.
 * The residual is the scalar distance from the transformed source point to the plane
 * defined by the target point and its normal.
 */
struct PointToPlaneError {
    const Vector3d source_point;
    const Vector3d target_point;
    const Vector3d target_normal;

    PointToPlaneError(const Vector3d& source, const Vector3d& target, const Vector3d& normal)
        : source_point(source), target_point(target), target_normal(normal) {}

    template <typename T>
    bool operator()(const T* const pose_increment, T* residuals) const {
        T source_point_T[3] = {T(source_point.x()), T(source_point.y()), T(source_point.z())};
        T transformed_point_T[3];

        // Apply the rotation and translation increment.
        ceres::AngleAxisRotatePoint(pose_increment, source_point_T, transformed_point_T);
        transformed_point_T[0] += pose_increment[3];
        transformed_point_T[1] += pose_increment[4];
        transformed_point_T[2] += pose_increment[5];

        // The difference vector between the target and the transformed source.
T diff[3] = {
    transformed_point_T[0] - T(target_point.x()),
    transformed_point_T[1] - T(target_point.y()),
    transformed_point_T[2] - T(target_point.z())
};
residuals[0] = T(target_normal.x()) * diff[0] +
               T(target_normal.y()) * diff[1] +
               T(target_normal.z()) * diff[2];

        return true;
    }
};


/**
 * @class LevenbergMarquardtICPOptimizer
 * @brief An ICP implementation that uses the Ceres Solver library for optimization.
 */
class LevenbergMarquardtICPOptimizer : public ICPOptimizer {
public:
    LevenbergMarquardtICPOptimizer() {
        // Ceres options are configured directly during the solve step,
        // so fewer parameters are needed here compared to a manual implementation.
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) override {
        // For precision with Ceres, we will work with doubles.
        Matrix4d estimatedPose = initialPose.cast<double>();

        // Build the FLANN index for the target point cloud for fast nearest neighbor search.
        m_nearestNeighborSearch->buildIndex(target.getPoints());

        for (int i = 0; i < m_nIterations; ++i) {
            std::cout << "=== ICP Iteration " << (i + 1) << "/" << m_nIterations << " ===" << std::endl;
            ceres::Problem problem;

            // 1. Transform source points with the current estimated pose to find correspondences.
            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose.cast<float>());
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose.cast<float>());

            // 2. Find correspondences using the nearest neighbor search.
            std::cout << "Matching points..." << std::endl;
            auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            pruneCorrespondences(transformedNormals, target.getNormals(), matches);
            std::cout << "Matching complete." << std::endl;

            // 3. Set up and solve the non-linear least squares problem with Ceres.
                    // The parameter to optimize is a 6-DoF pose increment (angle-axis rotation + translation).
            // We start with an increment of zero for each ICP iteration.
            double pose_increment[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            
            int validCorrespondences = 0;
            for (size_t j = 0; j < transformedPoints.size(); ++j) {
                if (matches[j].idx < 0) continue; // Skip invalid matches

                validCorrespondences++;
                // *** CRITICAL FIX: ***
                // The optimization should refine the current pose. Therefore, the cost function
                // needs to be built using the points that were actually matched: the *already transformed*
                // source points. The solver will then find a small `pose_increment` to apply to these
                // transformed points to better align them with their targets.
                const Vector3f& source_pt_f = transformedPoints[j]; 
                const Vector3f& target_pt_f = target.getPoints()[matches[j].idx];
                Vector3f target_normal_f = target.getNormals()[matches[j].idx]; 

                // Convert to double for Ceres
                Vector3d source_pt = source_pt_f.cast<double>();
                Vector3d target_pt = target_pt_f.cast<double>();
                Vector3d target_normal = target_normal_f.cast<double>();

                // *** POINT-TO-PLANE FIX: ***
                // The point-to-plane error metric requires the normal to be a unit vector.
                // We normalize it here to ensure the cost function is correctly scaled.
                if (m_bUsePointToPlaneConstraints) {
                    target_normal.normalize();
                }
                ceres::CostFunction* cost_function;
                if (m_bUsePointToPlaneConstraints) {
                    

                    cost_function = PointToPlaneConstraint::create(
                        source_pt_f,
                        target_pt_f,
                        target_normal_f,
                        1.0f // Weight can be adjusted or made adaptive.
                    );
                } else {
                    // Keep using PointToPointError if needed
                     cost_function = new ceres::AutoDiffCostFunction<PointToPointError, 3, 6>(
                        new PointToPointError(source_pt, target_pt)
                    );
                }
                
                // *** ROBUSTNESS IMPROVEMENT: ***
                // Use a loss function to reduce the influence of outlier correspondences.
                // CauchyLoss is a good general-purpose choice.
                ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
                problem.AddResidualBlock(cost_function, loss_function, pose_increment);
               
            }
            
            std::cout << "Valid correspondences: " << validCorrespondences << "/" << transformedPoints.size() << std::endl;
            if (validCorrespondences < 10) { // Need a minimum number of points to be stable
                std::cerr << "Not enough valid correspondences. Stopping ICP." << std::endl;
                break;
            }

            // 4. Configure and run the solver.
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = 50;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << "\n";
            
            // 5. Convert the resulting increment to a matrix and pre-multiply it to the current pose.
            Matrix4d increment_matrix = convertIncrementToMatrix(pose_increment);
            estimatedPose = increment_matrix * estimatedPose;
            // ✅ Re-orthogonalize rotation to prevent drift

            Eigen::Matrix3d R = estimatedPose.block<3,3>(0,0);
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
            estimatedPose.block<3,3>(0,0) = svd.matrixU() * svd.matrixV().transpose(); // R ← closest SO(3)

                            
            // Output summary for this iteration
            double angle, tx, ty, tz;
            getIncrementAsAngleAndTranslation(pose_increment, angle, tx, ty, tz);
            std::cout << "Pose update - Translation: " << Vector3d(tx, ty, tz).norm() << ", Rotation: " << (angle * 180.0 / M_PI) << " deg" << std::endl;
            
            // Convergence check
            if (Vector3d(tx, ty, tz).norm() < 1e-4 && angle < 1e-4) {
                 std::cout << "Converged after " << i+1 << " iterations." << std::endl;
                 break;
            }
            std::cout << "Solver iters: " << summary.iterations.size()
          << ", init cost: " << summary.initial_cost
          << ", final cost: " << summary.final_cost
          << ", termination: " << summary.termination_type << "\n";

            std::cout << std::endl;
        }

        // Store final result back into the float matrix.
        initialPose = estimatedPose.cast<float>();
    }

private:
    /**
     * @brief Converts a 6-element pose increment (angle-axis, translation) to a 4x4 transformation matrix.
     */
    Matrix4d convertIncrementToMatrix(const double* const pose_increment) {
        Matrix3d rotation;
        // Convert the angle-axis vector to a rotation matrix.
        ceres::AngleAxisToRotationMatrix(pose_increment, rotation.data());

        Vector3d translation(pose_increment[3], pose_increment[4], pose_increment[5]);

        Matrix4d transform = Matrix4d::Identity();
        transform.block<3, 3>(0, 0) = rotation;
        transform.block<3, 1>(0, 3) = translation;

        return transform;
    }

    /**
     * @brief Helper to extract human-readable values from the pose increment.
     */
    void getIncrementAsAngleAndTranslation(const double* const pose_increment, double& angle, double& tx, double& ty, double& tz) {
        angle = sqrt(pose_increment[0]*pose_increment[0] + pose_increment[1]*pose_increment[1] + pose_increment[2]*pose_increment[2]);
        tx = pose_increment[3];
        ty = pose_increment[4];
        tz = pose_increment[5];
    }
};

class AlternatingSymmetricICPOptimizer : public ICPOptimizer {
public:
    AlternatingSymmetricICPOptimizer() {}
    ~AlternatingSymmetricICPOptimizer() = default;

    void estimatePose(const PointCloud& source, const PointCloud& target, Eigen::Matrix4f& transformation) override {
        Eigen::Matrix4f estimatedPose = transformation;

        for (int i = 0; i < m_nIterations; ++i) {
            std::cout << "Alternating Symmetric ICP Iteration " << i + 1 << "/" << m_nIterations << std::endl;
            clock_t iterStart = clock();

            // --- STEP 1: Source-to-Target Optimization ---
            std::cout << "  Step 1: Optimizing S->T..." << std::endl;
            Eigen::Matrix4f deltaT1 = optimizeSourceToTarget(source, target, estimatedPose);
            estimatedPose = deltaT1 * estimatedPose;

            // --- STEP 2: Target-to-Source Optimization ---  
            std::cout << "  Step 2: Optimizing T->S..." << std::endl;
            Eigen::Matrix4f deltaT2 = optimizeTargetToSource(source, target, estimatedPose);
            // Apply inverse since we optimized T->S but need S->T transformation
            estimatedPose = deltaT2.inverse() * estimatedPose;

            clock_t iterEnd = clock();
            double elapsedSecs = double(iterEnd - iterStart) / CLOCKS_PER_SEC;
            std::cout << "  Iteration " << i + 1 << " completed in " << elapsedSecs << " seconds." << std::endl;
            std::cout << "  Current estimated pose:" << std::endl;
            std::cout << estimatedPose << std::endl;
        }

        transformation = estimatedPose;
    }

private:
    // Optimize Source-to-Target: Find transformation that moves source points to target
    Eigen::Matrix4f optimizeSourceToTarget(const PointCloud& source, const PointCloud& target, 
                                          const Eigen::Matrix4f& currentPose) {
        // Transform source points with current pose
        auto transformedSourcePoints = transformPoints(source.getPoints(), currentPose);
        auto transformedSourceNormals = transformNormals(source.getNormals(), currentPose);

        // Find correspondences: transformed source -> target
        m_nearestNeighborSearch->buildIndex(target.getPoints());
        auto matches = m_nearestNeighborSearch->queryMatches(transformedSourcePoints);
        
        // Prune correspondences based on normal compatibility
        pruneCorrespondences(transformedSourceNormals, target.getNormals(), matches);

        // Collect valid correspondences
        std::vector<Eigen::Vector3f> sourcePoints, targetPoints;
        std::vector<Eigen::Vector3f> sourceNormals, targetNormals;
        
        for (size_t j = 0; j < transformedSourcePoints.size(); ++j) {
            const auto& match = matches[j];
            if (match.idx >= 0) {
                sourcePoints.push_back(transformedSourcePoints[j]);
                sourceNormals.push_back(transformedSourceNormals[j]);
                targetPoints.push_back(target.getPoints()[match.idx]);
                targetNormals.push_back(target.getNormals()[match.idx]);
            }
        }

        std::cout << "    S->T correspondences: " << sourcePoints.size() << std::endl;

        if (sourcePoints.empty()) {
            std::cout << "    No S->T correspondences found, returning identity." << std::endl;
            return Eigen::Matrix4f::Identity();
        }

        // Optimize pose increment
        if (m_bUsePointToPlaneConstraints) {
            return estimatePoseNonLinear(sourcePoints, targetPoints, sourceNormals, targetNormals);
        } else {
            return estimatePosePointToPoint(sourcePoints, targetPoints);
        }
    }

    // Optimize Target-to-Source: Find transformation that moves target points to source
    Eigen::Matrix4f optimizeTargetToSource(const PointCloud& source, const PointCloud& target, 
                                          const Eigen::Matrix4f& currentPose) {
        // Transform source points with current pose (this is our "reference" now)
        auto transformedSourcePoints = transformPoints(source.getPoints(), currentPose);
        auto transformedSourceNormals = transformNormals(source.getNormals(), currentPose);

        // Find correspondences: target -> transformed source
        m_nearestNeighborSearch->buildIndex(transformedSourcePoints);
        auto matches = m_nearestNeighborSearch->queryMatches(target.getPoints());
        
        // Prune correspondences based on normal compatibility
        pruneCorrespondences(target.getNormals(), transformedSourceNormals, matches);

        // Collect valid correspondences
        std::vector<Eigen::Vector3f> sourcePoints, targetPoints;
        std::vector<Eigen::Vector3f> sourceNormals, targetNormals;
        
        for (size_t j = 0; j < target.getPoints().size(); ++j) {
            const auto& match = matches[j];
            if (match.idx >= 0) {
                // For T->S optimization: target is "source", transformed source is "target"
                sourcePoints.push_back(target.getPoints()[j]);
                sourceNormals.push_back(target.getNormals()[j]);
                targetPoints.push_back(transformedSourcePoints[match.idx]);
                targetNormals.push_back(transformedSourceNormals[match.idx]);
            }
        }

        std::cout << "    T->S correspondences: " << sourcePoints.size() << std::endl;

        if (sourcePoints.empty()) {
            std::cout << "    No T->S correspondences found, returning identity." << std::endl;
            return Eigen::Matrix4f::Identity();
        }

        // Optimize pose increment (this gives us the T->S transformation)
        if (m_bUsePointToPlaneConstraints) {
            return estimatePoseNonLinear(sourcePoints, targetPoints, sourceNormals, targetNormals);
        } else {
            return estimatePosePointToPoint(sourcePoints, targetPoints);
        }
    }

    Eigen::Matrix4f estimatePosePointToPoint(const std::vector<Eigen::Vector3f>& sourcePoints, 
                                           const std::vector<Eigen::Vector3f>& targetPoints) {
        ProcrustesAligner procrustAligner;
        return procrustAligner.estimatePose(sourcePoints, targetPoints);
    }

    Eigen::Matrix4f estimatePoseNonLinear(const std::vector<Eigen::Vector3f>& sourcePoints,
                                         const std::vector<Eigen::Vector3f>& targetPoints,
                                         const std::vector<Eigen::Vector3f>& sourceNormals,
                                         const std::vector<Eigen::Vector3f>& targetNormals) {
        
        // Use regular point-to-plane constraints (not symmetric)
        double pose_increment[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        ceres::Problem problem;

        for (size_t i = 0; i < sourcePoints.size(); ++i) {
            if (i >= targetNormals.size()) continue;

            // Use standard point-to-plane constraint
            ceres::CostFunction* cost_function =
                PointToPlaneConstraint::create(
                    sourcePoints[i],
                    targetPoints[i],
                    targetNormals[i],  // Use target normal for the plane
                    1.0
                );

            problem.AddResidualBlock(cost_function,
                                   new ceres::HuberLoss(0.1),
                                   pose_increment);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false; // Reduce output
        options.max_num_iterations = 25;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.num_threads = 4;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Convert result to transformation matrix
        Eigen::Matrix4f deltaPose = Eigen::Matrix4f::Identity();
        
        Eigen::Vector3d angle_axis_vec(pose_increment[0], pose_increment[1], pose_increment[2]);
        Eigen::Matrix3d rotation_matrix_double;
        ceres::AngleAxisToRotationMatrix(angle_axis_vec.data(), rotation_matrix_double.data());

        deltaPose.block<3, 3>(0, 0) = rotation_matrix_double.cast<float>();
        deltaPose.block<3, 1>(0, 3) = Eigen::Vector3f(
            static_cast<float>(pose_increment[3]),
            static_cast<float>(pose_increment[4]),
            static_cast<float>(pose_increment[5])
        );

        return deltaPose;
    }
};

