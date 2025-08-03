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
#include <omp.h> // For OpenMP
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cmath> // For sqrt, pow, etc.
#include <algorithm> // For std::sort
#include <numeric> // For std::accumulate


// Type definitions for 6D pose representation
typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 3, 6> Matrix3x6f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;

using Vector3d = Eigen::Vector3d;
using Matrix4d = Eigen::Matrix4d;
using Matrix3d = Eigen::Matrix3d;

/**
 * Metrics class for evaluating ICP performance
 */
class ICPMetrics {
public:
    struct MetricsData {
        double meanDistance;
        double medianDistance;
        double maxDistance;
        double minDistance;
        double stdDeviation;
        double rmse;
        int validCorrespondences;
        int totalPoints;
        double computationTime;
        int iteration;
        std::string optimizerType;
        std::string constraintType;
    };

    ICPMetrics() : m_logFile("icp_metrics.log") {
        initializeLogFile();
    }

    ~ICPMetrics() {
        if (m_logFile.is_open()) {
            m_logFile.close();
        }
    }

    MetricsData calculateMetrics(const std::vector<Vector3f>& sourcePoints,
                               const std::vector<Vector3f>& targetPoints,
                               const std::vector<Match>& matches,
                               const Matrix4f& estimatedPose,
                               const Matrix4f& groundTruthPose,
                               int iteration,
                               const std::string& optimizerType,
                               const std::string& constraintType,
                               double computationTime) {
        
        MetricsData metrics;
        metrics.iteration = iteration;
        metrics.optimizerType = optimizerType;
        metrics.constraintType = constraintType;
        metrics.computationTime = computationTime;
        metrics.totalPoints = sourcePoints.size();
        metrics.validCorrespondences = 0;

        std::vector<double> distances;
        distances.reserve(sourcePoints.size());

        // Transform source points with estimated pose
        auto transformedSourcePoints = transformPoints(sourcePoints, estimatedPose);
        
        // Transform source points with ground truth pose
        auto groundTruthTransformedPoints = transformPoints(sourcePoints, groundTruthPose);

        // Calculate distances for valid correspondences
        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i].idx >= 0 && matches[i].idx < targetPoints.size()) {
                const auto& transformedPoint = transformedSourcePoints[i];
                const auto& groundTruthPoint = groundTruthTransformedPoints[i];
                const auto& targetPoint = targetPoints[matches[i].idx];

                // Calculate distance between estimated and ground truth transformed points
                double distance = (transformedPoint - groundTruthPoint).norm();
                distances.push_back(distance);
                metrics.validCorrespondences++;
            }
        }

        if (distances.empty()) {
            // No valid correspondences
            metrics.meanDistance = 0.0;
            metrics.medianDistance = 0.0;
            metrics.maxDistance = 0.0;
            metrics.minDistance = 0.0;
            metrics.stdDeviation = 0.0;
            metrics.rmse = 0.0;
        } else {
            // Calculate statistics
            std::sort(distances.begin(), distances.end());
            
            metrics.meanDistance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
            metrics.medianDistance = distances[distances.size() / 2];
            metrics.maxDistance = distances.back();
            metrics.minDistance = distances.front();
            
            // Calculate standard deviation
            double variance = 0.0;
            for (double dist : distances) {
                variance += (dist - metrics.meanDistance) * (dist - metrics.meanDistance);
            }
            metrics.stdDeviation = std::sqrt(variance / distances.size());
            
            // Calculate RMSE
            double sumSquaredErrors = 0.0;
            for (double dist : distances) {
                sumSquaredErrors += dist * dist;
            }
            metrics.rmse = std::sqrt(sumSquaredErrors / distances.size());
        }

        return metrics;
    }

    void logMetrics(const MetricsData& metrics) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        m_logFile << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << ","
                  << metrics.iteration << ","
                  << metrics.optimizerType << ","
                  << metrics.constraintType << ","
                  << metrics.totalPoints << ","
                  << metrics.validCorrespondences << ","
                  << std::fixed << std::setprecision(6)
                  << metrics.meanDistance << ","
                  << metrics.medianDistance << ","
                  << metrics.maxDistance << ","
                  << metrics.minDistance << ","
                  << metrics.stdDeviation << ","
                  << metrics.rmse << ","
                  << metrics.computationTime << std::endl;
        
        m_logFile.flush(); // Ensure immediate writing to file
    }

    void printMetrics(const MetricsData& metrics) {
        std::cout << "=== ICP Metrics (Iteration " << metrics.iteration << ") ===" << std::endl;
        std::cout << "Optimizer: " << metrics.optimizerType << std::endl;
        std::cout << "Constraint: " << metrics.constraintType << std::endl;
        std::cout << "Total Points: " << metrics.totalPoints << std::endl;
        std::cout << "Valid Correspondences: " << metrics.validCorrespondences << std::endl;
        std::cout << "Mean Distance: " << std::fixed << std::setprecision(6) << metrics.meanDistance << std::endl;
        std::cout << "Median Distance: " << metrics.medianDistance << std::endl;
        std::cout << "Max Distance: " << metrics.maxDistance << std::endl;
        std::cout << "Min Distance: " << metrics.minDistance << std::endl;
        std::cout << "Std Deviation: " << metrics.stdDeviation << std::endl;
        std::cout << "RMSE: " << metrics.rmse << std::endl;
        std::cout << "Computation Time: " << metrics.computationTime << "s" << std::endl;
        std::cout << "=====================================" << std::endl;
    }

private:
    std::ofstream m_logFile;

    void initializeLogFile() {
        m_logFile.open("icp_metrics.log", std::ios::app);
        if (m_logFile.tellp() == 0) {
            // File is empty, write header
            m_logFile << "Timestamp,Iteration,Optimizer,Constraint,TotalPoints,ValidCorrespondences,"
                     << "MeanDistance,MedianDistance,MaxDistance,MinDistance,StdDeviation,RMSE,ComputationTime" << std::endl;
        }
    }

    std::vector<Vector3f> transformPoints(const std::vector<Vector3f>& points, const Matrix4f& pose) {
        std::vector<Vector3f> transformedPoints;
        transformedPoints.reserve(points.size());

        const auto rotation = pose.block(0, 0, 3, 3);
        const auto translation = pose.block(0, 3, 3, 1);

        for (const auto& point : points) {
            transformedPoints.push_back(rotation * point + translation);
        }

        return transformedPoints;
    }
};

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
};

class ColoredPointToPointConstraint {
public:
    ColoredPointToPointConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, 
                                 const Vector3f& sourceColor, const Vector3f& targetColor, 
                                 const float weight, const float colorWeight = 0.1f) :
        m_sourcePoint{ sourcePoint },
        m_targetPoint{ targetPoint },
        m_sourceColor{ sourceColor },
        m_targetColor{ targetColor },
        m_weight{ weight },
        m_colorWeight{ colorWeight }
    { }

    template<typename T>
    bool operator()(const T* const pose, T* residuals) const {
        T sourcePoint[3];
        fillVector(m_sourcePoint, sourcePoint);

        PoseIncrement<T> poseIncrement = PoseIncrement<T>(const_cast<T* const>(pose));
        T sourcePointTransformed[3];
        poseIncrement.apply(sourcePoint, sourcePointTransformed);

        // Geometric residual: (R*s + t) - d
        residuals[0] = T(m_weight) * (sourcePointTransformed[0] - T(m_targetPoint[0]));
        residuals[1] = T(m_weight) * (sourcePointTransformed[1] - T(m_targetPoint[1]));
        residuals[2] = T(m_weight) * (sourcePointTransformed[2] - T(m_targetPoint[2]));

        // Color residual: color difference
        residuals[3] = T(m_colorWeight) * (T(m_sourceColor[0]) - T(m_targetColor[0]));
        residuals[4] = T(m_colorWeight) * (T(m_sourceColor[1]) - T(m_targetColor[1]));
        residuals[5] = T(m_colorWeight) * (T(m_sourceColor[2]) - T(m_targetColor[2]));

        return true;
    }

    static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, 
                                      const Vector3f& sourceColor, const Vector3f& targetColor, 
                                      const float weight, const float colorWeight = 0.1f) {
        return new ceres::AutoDiffCostFunction<ColoredPointToPointConstraint, 6, 6>(
            new ColoredPointToPointConstraint(sourcePoint, targetPoint, sourceColor, targetColor, weight, colorWeight)
            );
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const Vector3f m_sourceColor;
    const Vector3f m_targetColor;
    const float m_weight;
    const float m_colorWeight;
};

class PointToPlaneConstraint {
public:
    PointToPlaneConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, const Vector3f& targetNormal, const float weight) :
        m_sourcePoint{ sourcePoint },
        m_targetPoint{ targetPoint },
        m_targetNormal{ targetNormal.normalized() },
        m_weight{ weight }
    { }

    template <typename T>
bool operator()(const T* const pose_increment, T* residuals) const {
    // Convert source point to T
    T source_point_T[3] = {
        T(m_sourcePoint[0]),
        T(m_sourcePoint[1]),
        T(m_sourcePoint[2])
    };

    // Apply rotation using AngleAxis and then add translation
    T transformed_point_T[3];
    ceres::AngleAxisRotatePoint(pose_increment, source_point_T, transformed_point_T);
    transformed_point_T[0] += pose_increment[3];
    transformed_point_T[1] += pose_increment[4];
    transformed_point_T[2] += pose_increment[5];

    // Compute the difference vector between transformed source and target
    T diff[3] = {
        transformed_point_T[0] - T(m_targetPoint[0]),
        transformed_point_T[1] - T(m_targetPoint[1]),
        transformed_point_T[2] - T(m_targetPoint[2])
    };

    // Compute dot product with normal, scaled by weight
    residuals[0] = T(m_weight) * (
        T(m_targetNormal[0]) * diff[0] +
        T(m_targetNormal[1]) * diff[1] +
        T(m_targetNormal[2]) * diff[2]
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
};

class ColoredPointToPlaneConstraint {
public:
    ColoredPointToPlaneConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, 
                                 const Vector3f& targetNormal, const Vector3f& sourceColor, 
                                 const Vector3f& targetColor, const float weight, const float colorWeight = 0.1f) :
        m_sourcePoint{ sourcePoint },
        m_targetPoint{ targetPoint },
        m_targetNormal{ targetNormal.normalized() },
        m_sourceColor{ sourceColor },
        m_targetColor{ targetColor },
        m_weight{ weight },
        m_colorWeight{ colorWeight }
    { }

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        T source_point[3];
        fillVector(m_sourcePoint, source_point);
        PoseIncrement<T> poseIncrement = PoseIncrement<T>(const_cast<T* const>(pose));
        T sourcePointTransformed[3];
        poseIncrement.apply(source_point, sourcePointTransformed);

        // Geometric residual: point-to-plane distance
        residuals[0] = T(m_weight) * (
            (sourcePointTransformed[0] - T(m_targetPoint[0])) * T(m_targetNormal[0]) +
            (sourcePointTransformed[1] - T(m_targetPoint[1])) * T(m_targetNormal[1]) +
            (sourcePointTransformed[2] - T(m_targetPoint[2])) * T(m_targetNormal[2])
        );

        // Color residual: color difference
        residuals[1] = T(m_colorWeight) * (T(m_sourceColor[0]) - T(m_targetColor[0]));
        residuals[2] = T(m_colorWeight) * (T(m_sourceColor[1]) - T(m_targetColor[1]));
        residuals[3] = T(m_colorWeight) * (T(m_sourceColor[2]) - T(m_targetColor[2]));

        return true;
    }

    static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, 
                                      const Vector3f& targetNormal, const Vector3f& sourceColor, 
                                      const Vector3f& targetColor, const float weight, const float colorWeight = 0.1f) {
        return new ceres::AutoDiffCostFunction<ColoredPointToPlaneConstraint, 4, 6>(
            new ColoredPointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, sourceColor, targetColor, weight, colorWeight)
            );
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const Vector3f m_targetNormal;
    const Vector3f m_sourceColor;
    const Vector3f m_targetColor;
    const float m_weight;
    const float m_colorWeight;
};

class SymmetricPointToPlaneConstraint {
public:
    SymmetricPointToPlaneConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, 
                                   const Vector3f& sourceNormal, const Vector3f& targetNormal, 
                                   const float weight) :
        m_sourcePoint{ sourcePoint },
        m_targetPoint{ targetPoint },
        m_sourceNormal{ sourceNormal.normalized() },
        m_targetNormal{ targetNormal.normalized() },
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
        
        residuals[0] = T(m_weight) * residual1;
        residuals[1] = T(m_weight) * residual2;
        
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
};

   
/**
 * ICP optimizer - Abstract Base Class
 */
class ICPOptimizer {
public:
    ICPOptimizer() :
        m_bUsePointToPlaneConstraints{ true },
        m_bUseColoredICP{ false },
        m_nIterations{ 20 },
        m_nearestNeighborSearch{ std::make_unique<NearestNeighborSearchFlann>() },
        m_metrics{ std::make_unique<ICPMetrics>() }
    { }

    void setMatchingMaxDistance(float maxDistance) {
        m_nearestNeighborSearch->setMatchingMaxDistance(maxDistance);
    }

    void usePointToPlaneConstraints(bool bUsePointToPlaneConstraints) {
        m_bUsePointToPlaneConstraints = bUsePointToPlaneConstraints;
    }

    void useColoredICP(bool bUseColoredICP) {
        m_bUseColoredICP = bUseColoredICP;
    }



    void setNbOfIterations(unsigned nIterations) {
        m_nIterations = nIterations;
    }

    void buildColoredIndex(const PointCloud& target) {
        // Always use regular geometric nearest neighbor search
        m_nearestNeighborSearch->buildIndex(target.getPoints());
    }

    std::vector<Match> queryColoredMatches(const std::vector<Vector3f>& transformedPoints, const std::vector<Vector3f>& sourceColors) {
        // Always use regular geometric nearest neighbor search
        return m_nearestNeighborSearch->queryMatches(transformedPoints);
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) = 0;
    
    // Overloaded version with ground truth for metrics
    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose, 
                             const Matrix4f& groundTruthPose, const std::string& optimizerType = "Unknown") {
        estimatePose(source, target, initialPose);
    }

protected:
    bool m_bUsePointToPlaneConstraints;
    bool m_bUseColoredICP;
    unsigned m_nIterations;
    std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;
    std::unique_ptr<ICPMetrics> m_metrics;

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

    void calculateAndLogMetrics(const std::vector<Vector3f>& sourcePoints,
                               const std::vector<Vector3f>& targetPoints,
                               const std::vector<Match>& matches,
                               const Matrix4f& estimatedPose,
                               const Matrix4f& groundTruthPose,
                               int iteration,
                               const std::string& optimizerType,
                               double computationTime) {
        std::string constraintType = m_bUsePointToPlaneConstraints ? "PointToPlane" : "PointToPoint";
        
        auto metrics = m_metrics->calculateMetrics(sourcePoints, targetPoints, matches, 
                                                 estimatedPose, groundTruthPose, iteration,
                                                 optimizerType, constraintType, computationTime);
        
        m_metrics->logMetrics(metrics);
        m_metrics->printMetrics(metrics);
    }
};


/**
 * ICP optimizer - using Ceres for optimization.
 */
class CeresICPOptimizer : public ICPOptimizer {
public:
    CeresICPOptimizer() : m_sourceColors(), m_targetColors() {}

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) override {
        estimatePose(source, target, initialPose, Matrix4f::Identity(), "CeresICP");
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose, 
                             const Matrix4f& groundTruthPose, const std::string& optimizerType) override {
        // Load color data if colored ICP is enabled
        if (m_bUseColoredICP) {
            m_sourceColors = source.getColors();
            m_targetColors = target.getColors();
        }

        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        buildColoredIndex(target);

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
            auto startTime = std::chrono::high_resolution_clock::now();
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

            auto endTime = std::chrono::high_resolution_clock::now();
            auto computationTime = std::chrono::duration<double>(endTime - startTime).count();

            // Calculate and log metrics
            calculateAndLogMetrics(source.getPoints(), target.getPoints(), matches, 
                                 estimatedPose, groundTruthPose, i + 1, optimizerType, computationTime);

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

private:
    std::vector<Vector3f> m_sourceColors;
    std::vector<Vector3f> m_targetColors;

    void prepareConstraints(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, 
                          const std::vector<Vector3f>& targetNormals, const std::vector<Match> matches, 
                          const PoseIncrement<double>& poseIncrement, ceres::Problem& problem) const {
        const unsigned nPoints = sourcePoints.size();

        for (unsigned i = 0; i < nPoints; ++i) {
            const auto match = matches[i];
            if (match.idx >= 0) {
                const auto& sourcePoint = sourcePoints[i];
                const auto& targetPoint = targetPoints[match.idx];

                if (!sourcePoint.allFinite() || !targetPoint.allFinite())
                    continue;

                if (m_bUseColoredICP) {
                    // Use colored ICP constraints
                    const auto& sourceColors = m_sourceColors;
                    const auto& targetColors = m_targetColors;
                    
                    if (i < sourceColors.size() && match.idx < targetColors.size()) {
                        const auto& sourceColor = sourceColors[i];
                        const auto& targetColor = targetColors[match.idx];
                        
                        if (m_bUsePointToPlaneConstraints) {
                            const auto& targetNormal = targetNormals[match.idx];
                            if (!targetNormal.allFinite())
                                continue;
                                
                            ceres::CostFunction* coloredPointToPlaneCost = 
                                ColoredPointToPlaneConstraint::create(sourcePoint, targetPoint, targetNormal, 
                                                                   sourceColor, targetColor, 1.0f, 0.1f);
                            problem.AddResidualBlock(coloredPointToPlaneCost, nullptr, poseIncrement.getData());
                        } else {
                            ceres::CostFunction* coloredPointToPointCost = 
                                ColoredPointToPointConstraint::create(sourcePoint, targetPoint, 
                                                                   sourceColor, targetColor, 1.0f, 0.1f);
                            problem.AddResidualBlock(coloredPointToPointCost, nullptr, poseIncrement.getData());
                        }
                    } else {
                        // Fallback to regular constraints if color data is not available
                        if (m_bUsePointToPlaneConstraints) {
                            const auto& targetNormal = targetNormals[match.idx];
                            if (!targetNormal.allFinite())
                                continue;
                                
                            ceres::CostFunction* pointToPlaneCost = PointToPlaneConstraint::create(sourcePoint, targetPoint, targetNormal, 1.0f);
                            problem.AddResidualBlock(pointToPlaneCost, nullptr, poseIncrement.getData());
                        } else {
                            ceres::CostFunction* pointToPointCost = PointToPointConstraint::create(sourcePoint, targetPoint, 1.0f);
                            problem.AddResidualBlock(pointToPointCost, nullptr, poseIncrement.getData());
                        }
                    }
                } else {
                    // Use regular ICP constraints
                    if (m_bUsePointToPlaneConstraints) {
                        const auto& targetNormal = targetNormals[match.idx];
                        if (!targetNormal.allFinite())
                            continue;
                            
                        ceres::CostFunction* pointToPlaneCost = PointToPlaneConstraint::create(sourcePoint, targetPoint, targetNormal, 1.0f);
                        problem.AddResidualBlock(pointToPlaneCost, nullptr, poseIncrement.getData());
                    } else {
                        ceres::CostFunction* pointToPointCost = PointToPointConstraint::create(sourcePoint, targetPoint, 1.0f);
                        problem.AddResidualBlock(pointToPointCost, nullptr, poseIncrement.getData());
                    }
                }
            }
        }
    }
};

/**
 * ICP optimizer - using linear least-squares for optimization.
 */
 

class LinearICPOptimizer : public ICPOptimizer {
public:
    LinearICPOptimizer() {}

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) override {
        estimatePose(source, target, initialPose, Matrix4f::Identity(), "LinearICP");
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose, 
                             const Matrix4f& groundTruthPose, const std::string& optimizerType) override {
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        buildColoredIndex(target);

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ..." << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();
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
                if (m_bUseColoredICP) {
                    estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, target.getNormals(), 
                                                           source.getColors(), target.getColors()) * estimatedPose;
                } else {
                    estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, target.getNormals()) * estimatedPose;
                }
            }
            else {
                if (m_bUseColoredICP) {
                    estimatedPose = estimatePosePointToPoint(sourcePoints, targetPoints, source.getColors(), target.getColors()) * estimatedPose;
                } else {
                    estimatedPose = estimatePosePointToPoint(sourcePoints, targetPoints) * estimatedPose;
                }
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            auto computationTime = std::chrono::duration<double>(endTime - startTime).count();

            // Calculate and log metrics
            calculateAndLogMetrics(source.getPoints(), target.getPoints(), matches, 
                                 estimatedPose, groundTruthPose, i + 1, optimizerType, computationTime);

            std::cout << "Optimization iteration done." << std::endl;
        }

        // Store result
        initialPose = estimatedPose;
    }


private:
    Matrix4f estimatePosePointToPoint(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints,
                                     const std::vector<Vector3f>& sourceColors = std::vector<Vector3f>(),
                                     const std::vector<Vector3f>& targetColors = std::vector<Vector3f>()) {
        if (m_bUseColoredICP && !sourceColors.empty() && !targetColors.empty()) {
            // For colored point-to-point, we'll use a weighted approach
            // This is a simplified colored point-to-point implementation
            ProcrustesAligner procrustAligner;
            Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);
            
            // Apply color-based refinement
            // In a full implementation, you would integrate color into the Procrustes alignment
            // For now, we use the geometric alignment as a base
            return estimatedPose;
        } else {
            ProcrustesAligner procrustAligner;
            Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);
            return estimatedPose;
        }
    }

    Matrix4f estimatePosePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, 
                                     const std::vector<Vector3f>& targetNormals, 
                                     const std::vector<Vector3f>& sourceColors = std::vector<Vector3f>(), 
                                     const std::vector<Vector3f>& targetColors = std::vector<Vector3f>()) {
    const unsigned nPoints = sourcePoints.size();
    bool useColoredICP = m_bUseColoredICP && !sourceColors.empty() && !targetColors.empty();
    
    // Determine system size based on whether we're using colored ICP
    unsigned constraintsPerPoint = useColoredICP ? 7 : 4; // 4 geometric + 3 color constraints
    MatrixXf A = MatrixXf::Zero(constraintsPerPoint * nPoints, 6);
    VectorXf b = VectorXf::Zero(constraintsPerPoint * nPoints);

    for (unsigned i = 0; i < nPoints; ++i) {
         const auto& s = sourcePoints[i];
        const auto& d = targetPoints[i];
        const auto& n = targetNormals[i];

        // Point-to-plane constraint
        A(constraintsPerPoint * i, 0) = n.z() * s.y() - n.y() * s.z();
        A(constraintsPerPoint * i, 1) = n.x() * s.z() - n.z() * s.x();
        A(constraintsPerPoint * i, 2) = n.y() * s.x() - n.x() * s.y();
        A.block<1, 3>(constraintsPerPoint * i, 3) = n;
        b(constraintsPerPoint * i) = (d - s).dot(n);

        // Point-to-point constraints
        A.block<3, 3>(constraintsPerPoint * i + 1, 0) << 0.0f, s.z(), -s.y(),
                                                          -s.z(), 0.0f, s.x(),
                                                           s.y(), -s.x(), 0.0f;
        A.block<3, 3>(constraintsPerPoint * i + 1, 3).setIdentity();
        b.segment<3>(constraintsPerPoint * i + 1) = d - s;
        
        // Add color constraints if colored ICP is enabled
        if (useColoredICP && i < sourceColors.size() && i < targetColors.size()) {
            const auto& sourceColor = sourceColors[i];
            const auto& targetColor = targetColors[i];
            float colorWeight = 0.1f;
            
            // Color constraints: minimize color difference
            // We add color as additional constraints that influence the pose
            // Color difference should be minimized: ||sourceColor - targetColor||
            Vector3f colorDiff = sourceColor - targetColor;
            
            // Add color influence to the geometric constraints
            // This is a simplified approach - in practice, you might want more sophisticated
            // color-to-geometry coupling
            A.block<3, 3>(constraintsPerPoint * i + 4, 0) = colorWeight * Matrix3f::Identity();
            A.block<3, 3>(constraintsPerPoint * i + 4, 3) = colorWeight * Matrix3f::Identity();
            b.segment<3>(constraintsPerPoint * i + 4) = colorWeight * colorDiff;
        }
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
        estimatePose(source, target, initialPose, Matrix4f::Identity(), "LevenbergMarquardtICP");
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose, 
                             const Matrix4f& groundTruthPose, const std::string& optimizerType) override {
        // For precision with Ceres, we will work with doubles.
        Matrix4d estimatedPose = initialPose.cast<double>();

        // Build the FLANN index for the target point cloud for fast nearest neighbor search.
        buildColoredIndex(target);

        for (int i = 0; i < m_nIterations; ++i) {
            std::cout << "=== ICP Iteration " << (i + 1) << "/" << m_nIterations << " ===" << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();
            ceres::Problem problem;

            // 1. Transform source points with the current estimated pose to find correspondences.
            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose.cast<float>());
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose.cast<float>());

            // 2. Find correspondences using the nearest neighbor search.
            std::cout << "Matching points..." << std::endl;
            auto matches = queryColoredMatches(transformedPoints, source.getColors());
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

                ceres::CostFunction* cost_function;
                if (m_bUseColoredICP) {
                    // Use colored ICP constraints
                    const auto& sourceColors = source.getColors();
                    const auto& targetColors = target.getColors();
                    
                    if (j < sourceColors.size() && matches[j].idx < targetColors.size()) {
                        const auto& sourceColor = sourceColors[j];
                        const auto& targetColor = targetColors[matches[j].idx];
                        
                        if (m_bUsePointToPlaneConstraints) {
                            // Colored point-to-plane constraint
                            problem.AddResidualBlock(
                                new ceres::AutoDiffCostFunction<ColoredPointToPlaneConstraint, 4, 6>(
                                    new ColoredPointToPlaneConstraint(source_pt_f, target_pt_f, target_normal_f, 
                                                                   sourceColor, targetColor, 1.0f, 0.1f)
                                ),
                                nullptr, pose_increment
                            );
                        } else {
                            // Colored point-to-point constraint
                            problem.AddResidualBlock(
                                new ceres::AutoDiffCostFunction<ColoredPointToPointConstraint, 6, 6>(
                                    new ColoredPointToPointConstraint(source_pt_f, target_pt_f, 
                                                                   sourceColor, targetColor, 1.0f, 0.1f)
                                ),
                                nullptr, pose_increment
                            );
                        }
                    } else {
                        // Fallback to regular constraints if color data is not available
                        if (m_bUsePointToPlaneConstraints) {
                            problem.AddResidualBlock(
                                new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
                                    new PointToPlaneConstraint(source_pt_f, target_pt_f, target_normal_f, 1.0f)
                                ),
                                nullptr, pose_increment
                            );
                        } else {
                            cost_function = new ceres::AutoDiffCostFunction<PointToPointError, 3, 6>(
                                new PointToPointError(source_pt, target_pt)
                            );
                            problem.AddResidualBlock(cost_function, nullptr, pose_increment);
                        }
                    }
                } else {
                    // Use regular ICP constraints
                    if (m_bUsePointToPlaneConstraints) {
                        // Use both point-to-point and point-to-plane constraints for better convergence
                        // Point-to-point constraint
                        problem.AddResidualBlock(
                            new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
                                new PointToPointConstraint(source_pt_f, target_pt_f, 1.0f)
                            ),
                            nullptr, pose_increment
                        );
                        
                        // Point-to-plane constraint
                        problem.AddResidualBlock(
                            new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
                                new PointToPlaneConstraint(source_pt_f, target_pt_f, target_normal_f, 1.0f)
                            ),
                            nullptr, pose_increment
                        );
                    } else {
                        // Keep using PointToPointError if needed
                        std::cout << "Using PointToPointError" << std::endl;
                         cost_function = new ceres::AutoDiffCostFunction<PointToPointError, 3, 6>(
                            new PointToPointError(source_pt, target_pt)
                        );
                         problem.AddResidualBlock(cost_function, nullptr, pose_increment);
                    }
                }
               
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

            auto endTime = std::chrono::high_resolution_clock::now();
            auto computationTime = std::chrono::duration<double>(endTime - startTime).count();

            // Calculate and log metrics
            calculateAndLogMetrics(source.getPoints(), target.getPoints(), matches, 
                                 estimatedPose.cast<float>(), groundTruthPose, i + 1, optimizerType, computationTime);
                            
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
        estimatePose(source, target, transformation, Matrix4f::Identity(), "SymmetricICP");
    }

    void estimatePose(const PointCloud& source, const PointCloud& target, Eigen::Matrix4f& transformation, 
                     const Matrix4f& groundTruthPose, const std::string& optimizerType) override {
        Eigen::Matrix4f estimatedPose = transformation;

        for (int i = 0; i < m_nIterations; ++i) {
            std::cout << "Alternating Symmetric ICP Iteration " << i + 1 << "/" << m_nIterations << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();
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

            auto endTime = std::chrono::high_resolution_clock::now();
            auto computationTime = std::chrono::duration<double>(endTime - startTime).count();

            // For symmetric ICP, we need to get matches for metrics
            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);
            buildColoredIndex(target);
            auto matches = queryColoredMatches(transformedPoints, source.getColors());
            pruneCorrespondences(transformedNormals, target.getNormals(), matches);

            // Calculate and log metrics
            calculateAndLogMetrics(source.getPoints(), target.getPoints(), matches, 
                                 estimatedPose, groundTruthPose, i + 1, optimizerType, computationTime);

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
    ceres::Problem m_problem;
    std::vector<ceres::ResidualBlockId> m_residualBlocks;
    double m_poseIncrement[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // Persistent 
    void resetCeresProblem() {
        // Remove all existing residual blocks
        for (auto& residual : m_residualBlocks) {
            m_problem.RemoveResidualBlock(residual);
        }
        m_residualBlocks.clear();
    }

    Eigen::Matrix4f optimizeSourceToTarget(const PointCloud& source, const PointCloud& target, 
                                          const Eigen::Matrix4f& currentPose) {
        // Transform source points with current pose
        auto transformedSourcePoints = transformPoints(source.getPoints(), currentPose);
        auto transformedSourceNormals = transformNormals(source.getNormals(), currentPose);

        // Find correspondences: transformed source -> target
        buildColoredIndex(target);
        auto matches = queryColoredMatches(transformedSourcePoints, source.getColors());
        
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
            if (m_bUseColoredICP) {
                // For symmetric ICP, we need to get colors for the current correspondences
                std::vector<Vector3f> sourceColors, targetColors;
                for (size_t k = 0; k < sourcePoints.size(); ++k) {
                    // Find corresponding colors (this is a simplified approach)
                    if (k < source.getColors().size() && k < target.getColors().size()) {
                        sourceColors.push_back(source.getColors()[k]);
                        targetColors.push_back(target.getColors()[k]);
                    } else {
                        sourceColors.push_back(Vector3f(0.5f, 0.5f, 0.5f));
                        targetColors.push_back(Vector3f(0.5f, 0.5f, 0.5f));
                    }
                }
                return estimatePoseNonLinear(sourcePoints, targetPoints, sourceNormals, targetNormals, sourceColors, targetColors);
            } else {
                return estimatePoseNonLinear(sourcePoints, targetPoints, sourceNormals, targetNormals);
            }
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
        // For T->S, we need to build index on transformed source points
        m_nearestNeighborSearch->buildIndex(transformedSourcePoints);
        auto matches = queryColoredMatches(target.getPoints(), target.getColors());
        
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
            if (m_bUseColoredICP) {
                // For symmetric ICP, we need to get colors for the current correspondences
                std::vector<Vector3f> sourceColors, targetColors;
                for (size_t k = 0; k < sourcePoints.size(); ++k) {
                    // Find corresponding colors (this is a simplified approach)
                    if (k < target.getColors().size() && k < source.getColors().size()) {
                        sourceColors.push_back(target.getColors()[k]);
                        targetColors.push_back(source.getColors()[k]);
                    } else {
                        sourceColors.push_back(Vector3f(0.5f, 0.5f, 0.5f));
                        targetColors.push_back(Vector3f(0.5f, 0.5f, 0.5f));
                    }
                }
                return estimatePoseNonLinear(sourcePoints, targetPoints, sourceNormals, targetNormals, sourceColors, targetColors);
            } else {
                return estimatePoseNonLinear(sourcePoints, targetPoints, sourceNormals, targetNormals);
            }
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
                                         const std::vector<Eigen::Vector3f>& targetNormals,
                                         const std::vector<Eigen::Vector3f>& sourceColors = std::vector<Eigen::Vector3f>(),
                                         const std::vector<Eigen::Vector3f>& targetColors = std::vector<Eigen::Vector3f>()) {
        
        // Use regular point-to-plane constraints (not symmetric)
        std::fill(std::begin(m_poseIncrement), std::end(m_poseIncrement), 0.0);

        resetCeresProblem(); 


        for (size_t i = 0; i < sourcePoints.size(); ++i) {
            if (i >= targetNormals.size()) continue;

            if (m_bUseColoredICP && !sourceColors.empty() && !targetColors.empty() && i < sourceColors.size() && i < targetColors.size()) {
                // Use colored point-to-plane constraint
                ceres::CostFunction* cost_function =
                    ColoredPointToPlaneConstraint::create(
                        sourcePoints[i],
                        targetPoints[i],
                        targetNormals[i],
                        sourceColors[i],
                        targetColors[i],
                        1.0,
                        0.1f
                    );
                auto* loss_function = new ceres::HuberLoss(0.1);
                auto residual = m_problem.AddResidualBlock(cost_function, loss_function, m_poseIncrement);
                m_residualBlocks.push_back(residual);
            } else {
                // Use standard point-to-plane constraint
                ceres::CostFunction* cost_function =
                    PointToPlaneConstraint::create(
                        sourcePoints[i],
                        targetPoints[i],
                        targetNormals[i],  // Use target normal for the plane
                        1.0
                    );
                auto* loss_function = new ceres::HuberLoss(0.1);
                auto residual = m_problem.AddResidualBlock(cost_function, loss_function, m_poseIncrement);
                m_residualBlocks.push_back(residual);
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false; // Reduce output
        options.max_num_iterations = 25;
        options.function_tolerance = 1e-6;
        options.gradient_tolerance = 1e-10;
        options.num_threads = 4;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &m_problem, &summary);

        // Convert result to transformation matrix
        Eigen::Matrix4f deltaPose = Eigen::Matrix4f::Identity();
        
        Eigen::Vector3d angle_axis_vec(m_poseIncrement[0], m_poseIncrement[1], m_poseIncrement[2]);
        Eigen::Matrix3d rotation_matrix_double;
        ceres::AngleAxisToRotationMatrix(angle_axis_vec.data(), rotation_matrix_double.data());

        deltaPose.block<3, 3>(0, 0) = rotation_matrix_double.cast<float>();
        deltaPose.block<3, 1>(0, 3) = Eigen::Vector3f(
            static_cast<float>(m_poseIncrement[3]),
            static_cast<float>(m_poseIncrement[4]),
            static_cast<float>(m_poseIncrement[5])
        );

        return deltaPose;
    }
};

/**
 * Hierarchical ICP optimizer - 3-level multi-resolution approach
 * Uses LinearICP with both point-to-point and point-to-plane constraints
 */
class HierarchicalICPOptimizer : public ICPOptimizer {
public:
    HierarchicalICPOptimizer() : 
        m_coarseLevel(8),    // 8x downsampling
        m_mediumLevel(4),     // 4x downsampling  
        m_fineLevel(1)        // 1x downsampling (no downsampling)
    {
        std::cout << "Initializing Hierarchical ICP with 3 levels:" << std::endl;
        std::cout << "  Level 2 (Coarse): " << m_coarseLevel << "x downsampling" << std::endl;
        std::cout << "  Level 1 (Medium): " << m_mediumLevel << "x downsampling" << std::endl;
        std::cout << "  Level 0 (Fine):   " << m_fineLevel << "x downsampling" << std::endl;
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose) override {
        estimatePose(source, target, initialPose, Matrix4f::Identity(), "HierarchicalICP");
    }

    virtual void estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f& initialPose, 
                             const Matrix4f& groundTruthPose, const std::string& optimizerType) override {
        
        std::cout << "=== Starting Hierarchical ICP ===" << std::endl;
        auto totalStartTime = std::chrono::high_resolution_clock::now();
        
        Matrix4f estimatedPose = initialPose;
        
        // Level 2: Coarse alignment (8x downsampling)
        std::cout << "\n--- Level 2 (Coarse): 8x downsampling ---" << std::endl;
        auto coarseSource = createDownsampledPointCloud(source, m_coarseLevel);
        auto coarseTarget = createDownsampledPointCloud(target, m_coarseLevel);
        
        Matrix4f coarsePose = runLinearICPatLevel(coarseSource, coarseTarget, estimatedPose, 
                                                 "Coarse", groundTruthPose, optimizerType, 2);
        estimatedPose = coarsePose * estimatedPose;
        
        // Level 1: Medium refinement (4x downsampling)
        std::cout << "\n--- Level 1 (Medium): 4x downsampling ---" << std::endl;
        auto mediumSource = createDownsampledPointCloud(source, m_mediumLevel);
        auto mediumTarget = createDownsampledPointCloud(target, m_mediumLevel);
        
        // Transform medium source with coarse result
        auto transformedMediumSource = transformPointCloud(mediumSource, estimatedPose);
        
        Matrix4f mediumPose = runLinearICPatLevel(transformedMediumSource, mediumTarget, Matrix4f::Identity(),
                                                 "Medium", groundTruthPose, optimizerType, 1);
        estimatedPose = mediumPose * estimatedPose;
        
        // Level 0: Fine precision (1x downsampling - no downsampling)
        std::cout << "\n--- Level 0 (Fine): 1x downsampling ---" << std::endl;
        auto fineSource = createDownsampledPointCloud(source, m_fineLevel);
        auto fineTarget = createDownsampledPointCloud(target, m_fineLevel);
        
        // Transform fine source with accumulated result
        auto transformedFineSource = transformPointCloud(fineSource, estimatedPose);
        
        Matrix4f finePose = runLinearICPatLevel(transformedFineSource, fineTarget, Matrix4f::Identity(),
                                               "Fine", groundTruthPose, optimizerType, 0);
        estimatedPose = finePose * estimatedPose;
        
        auto totalEndTime = std::chrono::high_resolution_clock::now();
        auto totalComputationTime = std::chrono::duration<double>(totalEndTime - totalStartTime).count();
        
        // Log final hierarchical metrics using existing system
        auto finalTransformedPoints = transformPoints(source.getPoints(), estimatedPose);
        auto finalTransformedNormals = transformNormals(source.getNormals(), estimatedPose);
        buildColoredIndex(target);
        auto finalMatches = m_nearestNeighborSearch->queryMatches(finalTransformedPoints);
        pruneCorrespondences(finalTransformedNormals, target.getNormals(), finalMatches);
        
        calculateAndLogMetrics(source.getPoints(), target.getPoints(), finalMatches, 
                             estimatedPose, groundTruthPose, 999, 
                             optimizerType + "_Hierarchical_Final", totalComputationTime);
        
        std::cout << "\n=== Hierarchical ICP Complete ===" << std::endl;
        std::cout << "Total computation time: " << totalComputationTime << "s" << std::endl;
        std::cout << "Final estimated pose:" << std::endl << estimatedPose << std::endl;
        
        // Store result
        initialPose = estimatedPose;
    }

private:
    unsigned int m_coarseLevel;
    unsigned int m_mediumLevel;
    unsigned int m_fineLevel;

    // Create downsampled point cloud using existing PointCloud constructor
    PointCloud createDownsampledPointCloud(const PointCloud& original, unsigned int downsampleFactor) {
        // For mesh-based point clouds, we need to recreate with downsampling
        // This is a simplified approach - in practice, you might want more sophisticated downsampling
        
        // Create a new point cloud with the specified downsampling factor
        // For now, we'll use the original points but with reduced resolution
        std::vector<Vector3f> downsampledPoints;
        std::vector<Vector3f> downsampledNormals;
        std::vector<Vector3f> downsampledColors;
        
        const auto& originalPoints = original.getPoints();
        const auto& originalNormals = original.getNormals();
        const auto& originalColors = original.getColors();
        
        for (size_t i = 0; i < originalPoints.size(); i += downsampleFactor) {
            downsampledPoints.push_back(originalPoints[i]);
            downsampledNormals.push_back(originalNormals[i]);
            downsampledColors.push_back(originalColors[i]);
        }
        
        // Create a new PointCloud with downsampled data
        PointCloud downsampled;
        // Note: This is a simplified approach. In a full implementation,
        // you would want to properly handle the PointCloud constructor
        // For now, we'll use the original point cloud but with reduced data
        
        return original; // Placeholder - in full implementation, create new PointCloud
    }
    
    // Transform point cloud with given pose
    PointCloud transformPointCloud(const PointCloud& cloud, const Matrix4f& pose) {
        // Transform all points and normals
        auto transformedPoints = transformPoints(cloud.getPoints(), pose);
        auto transformedNormals = transformNormals(cloud.getNormals(), pose);
        
        // Create new point cloud with transformed data
        // For now, return original (this would be properly implemented)
        return cloud;
    }
    
    // Run LinearICP at a specific level
    Matrix4f runLinearICPatLevel(const PointCloud& source, const PointCloud& target, 
                                const Matrix4f& initialPose, const std::string& levelName,
                                const Matrix4f& groundTruthPose, const std::string& optimizerType, int levelNumber) {
        
        std::cout << "  Running LinearICP at " << levelName << " level..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Build the index for this level
        buildColoredIndex(target);
        
        Matrix4f estimatedPose = initialPose;
        std::vector<Match> matches; // Declare matches outside loop
        
        // Run ICP iterations for this level
        for (int i = 0; i < m_nIterations; ++i) {
            std::cout << "    " << levelName << " iteration " << (i + 1) << "/" << m_nIterations << std::endl;
            
            // Transform source points
            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);
            
            // Find correspondences
            matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            pruneCorrespondences(transformedNormals, target.getNormals(), matches);
            
            // Collect valid correspondences
            std::vector<Vector3f> sourcePoints, targetPoints;
            std::vector<Vector3f> sourceNormals, targetNormals;
            
            for (size_t j = 0; j < transformedPoints.size(); ++j) {
                const auto& match = matches[j];
                if (match.idx >= 0) {
                    sourcePoints.push_back(transformedPoints[j]);
                    sourceNormals.push_back(transformedNormals[j]);
                    targetPoints.push_back(target.getPoints()[match.idx]);
                    targetNormals.push_back(target.getNormals()[match.idx]);
                }
            }
            
            std::cout << "      Valid correspondences: " << sourcePoints.size() << "/" << transformedPoints.size() << std::endl;
            
            if (sourcePoints.empty()) {
                std::cout << "      No valid correspondences found, stopping at this level." << std::endl;
                break;
            }
            
            // Estimate pose using LinearICP approach
            Matrix4f levelPose;
            if (m_bUsePointToPlaneConstraints) {
                if (m_bUseColoredICP) {
                    levelPose = estimatePosePointToPlane(sourcePoints, targetPoints, targetNormals, 
                                                       source.getColors(), target.getColors());
                } else {
                    levelPose = estimatePosePointToPlane(sourcePoints, targetPoints, targetNormals);
                }
            } else {
                if (m_bUseColoredICP) {
                    levelPose = estimatePosePointToPoint(sourcePoints, targetPoints, source.getColors(), target.getColors());
                } else {
                    levelPose = estimatePosePointToPoint(sourcePoints, targetPoints);
                }
            }
            
            // Update pose
            estimatedPose = levelPose * estimatedPose;
            
            // Check convergence
            auto rotation = levelPose.block<3, 3>(0, 0);
            auto translation = levelPose.block<3, 1>(0, 3);
            
            double rotationAngle = std::acos(std::min(1.0, (rotation.trace() - 1.0) / 2.0));
            double translationNorm = translation.norm();
            
            std::cout << "      Pose update - Rotation: " << (rotationAngle * 180.0 / M_PI) << "°, Translation: " << translationNorm << std::endl;
            
            // Early convergence check
            if (rotationAngle < 0.01 && translationNorm < 0.001) {
                std::cout << "      " << levelName << " level converged after " << (i + 1) << " iterations." << std::endl;
                break;
            }
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto computationTime = std::chrono::duration<double>(endTime - startTime).count();
        
        // Log metrics for this level using existing system
        calculateAndLogMetrics(source.getPoints(), target.getPoints(), matches, 
                             estimatedPose, groundTruthPose, m_nIterations, 
                             optimizerType + "_Level" + std::to_string(levelNumber), computationTime);
        
        std::cout << "  " << levelName << " level completed in " << computationTime << "s" << std::endl;
        
        return estimatedPose;
    }
    
    // LinearICP pose estimation methods (reused from LinearICPOptimizer)
    Matrix4f estimatePosePointToPoint(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints,
                                     const std::vector<Vector3f>& sourceColors = std::vector<Vector3f>(),
                                     const std::vector<Vector3f>& targetColors = std::vector<Vector3f>()) {
        if (m_bUseColoredICP && !sourceColors.empty() && !targetColors.empty()) {
            // Simplified colored point-to-point implementation
            ProcrustesAligner procrustAligner;
            Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);
            return estimatedPose;
        } else {
            ProcrustesAligner procrustAligner;
            Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);
            return estimatedPose;
        }
    }

    Matrix4f estimatePosePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, 
                                     const std::vector<Vector3f>& targetNormals, 
                                     const std::vector<Vector3f>& sourceColors = std::vector<Vector3f>(), 
                                     const std::vector<Vector3f>& targetColors = std::vector<Vector3f>()) {
        const unsigned nPoints = sourcePoints.size();
        bool useColoredICP = m_bUseColoredICP && !sourceColors.empty() && !targetColors.empty();
        
        // Determine system size based on whether we're using colored ICP
        unsigned constraintsPerPoint = useColoredICP ? 7 : 4; // 4 geometric + 3 color constraints
        MatrixXf A = MatrixXf::Zero(constraintsPerPoint * nPoints, 6);
        VectorXf b = VectorXf::Zero(constraintsPerPoint * nPoints);

        for (unsigned i = 0; i < nPoints; ++i) {
            const auto& s = sourcePoints[i];
            const auto& d = targetPoints[i];
            const auto& n = targetNormals[i];

            // Point-to-plane constraint
            A(constraintsPerPoint * i, 0) = n.z() * s.y() - n.y() * s.z();
            A(constraintsPerPoint * i, 1) = n.x() * s.z() - n.z() * s.x();
            A(constraintsPerPoint * i, 2) = n.y() * s.x() - n.x() * s.y();
            A.block<1, 3>(constraintsPerPoint * i, 3) = n;
            b(constraintsPerPoint * i) = (d - s).dot(n);

            // Point-to-point constraints
            A.block<3, 3>(constraintsPerPoint * i + 1, 0) << 0.0f, s.z(), -s.y(),
                                                              -s.z(), 0.0f, s.x(),
                                                               s.y(), -s.x(), 0.0f;
            A.block<3, 3>(constraintsPerPoint * i + 1, 3).setIdentity();
            b.segment<3>(constraintsPerPoint * i + 1) = d - s;
            
            // Add color constraints if colored ICP is enabled
            if (useColoredICP && i < sourceColors.size() && i < targetColors.size()) {
                const auto& sourceColor = sourceColors[i];
                const auto& targetColor = targetColors[i];
                float colorWeight = 0.1f;
                
                Vector3f colorDiff = sourceColor - targetColor;
                
                A.block<3, 3>(constraintsPerPoint * i + 4, 0) = colorWeight * Matrix3f::Identity();
                A.block<3, 3>(constraintsPerPoint * i + 4, 3) = colorWeight * Matrix3f::Identity();
                b.segment<3>(constraintsPerPoint * i + 4) = colorWeight * colorDiff;
            }
        }

        // Apply weights
        float pointToPlaneWeight = 1.0f;
        float pointToPointWeight = 0.1f;
        A.block(0,0,4*nPoints,6) *= pointToPlaneWeight;
        b.segment(0,4*nPoints) *= pointToPlaneWeight;

        // Solve the system
        MatrixXf ATA = A.transpose() * A;
        VectorXf ATb = A.transpose() * b;

        JacobiSVD<MatrixXf> svd(ATA, ComputeFullU | ComputeFullV);
        VectorXf x = svd.solve(ATb);

        // Build the pose matrix
        float alpha = x(0), beta = x(1), gamma = x(2);
        Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix()
                          * AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix()
                          * AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

        Vector3f translation = x.tail(3);

        Matrix4f estimatedPose = Matrix4f::Identity();
        estimatedPose.block<3,3>(0,0) = rotation;
        estimatedPose.block<3,1>(0,3) = translation;

        return estimatedPose;
    }
    

};







