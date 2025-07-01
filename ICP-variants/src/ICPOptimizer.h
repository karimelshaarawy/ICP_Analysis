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

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {
        // Create PoseIncrement object to handle the transformation
        PoseIncrement<T> poseIncrement(const_cast<T*>(pose));
        
        // Convert source point to templated type
        T sourcePoint[3];
        fillVector(m_sourcePoint, sourcePoint);
        
        // Convert target point to templated type
        T targetPoint[3];
        fillVector(m_targetPoint, targetPoint);
        
       
        T transformedSourcePoint[3];
        poseIncrement.apply(sourcePoint, transformedSourcePoint);
        
       
        T sqrtWeight = T(sqrt(m_weight));
        residuals[0] = sqrtWeight * (transformedSourcePoint[0] - targetPoint[0]);
        residuals[1] = sqrtWeight * (transformedSourcePoint[1] - targetPoint[1]);
        residuals[2] = sqrtWeight * (transformedSourcePoint[2] - targetPoint[2]);
        
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
    // 1. Create a helper object to apply the pose transformation
    PoseIncrement<T> poseIncrement(const_cast<T*>(pose));

    // 2. Define the source point, target point, and target normal as templated arrays
    T sourcePoint[3];
    fillVector(m_sourcePoint, sourcePoint);

    T targetPoint[3];
    fillVector(m_targetPoint, targetPoint);

    T targetNormal[3];
    fillVector(m_targetNormal, targetNormal);
    
    // 3. Apply the current pose transformation to the source point
    T transformedSourcePoint[3];
    poseIncrement.apply(sourcePoint, transformedSourcePoint);

    // 4. Calculate the difference vector between the transformed source and the target
    T diff[3];
    diff[0] = transformedSourcePoint[0] - targetPoint[0];
    diff[1] = transformedSourcePoint[1] - targetPoint[1];
    diff[2] = transformedSourcePoint[2] - targetPoint[2];

    // 5. Calculate the residual: the dot product of the difference vector and the target normal
    T dotProduct = diff[0] * targetNormal[0] + diff[1] * targetNormal[1] + diff[2] * targetNormal[2];

    // 6. Apply the weight
    T sqrtWeight = T(sqrt(m_weight));
    residuals[0] = sqrtWeight * dotProduct;

    return true;
}
    }

    static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, const Vector3f& targetNormal, const float weight) {
        return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
            new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, weight)
            );
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
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
        m_bUsePointToPlaneConstraints{ false },
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
        int prunedCount = 0;

        for (unsigned i = 0; i < nPoints; i++) {
            Match& match = matches[i];
            if (match.idx >= 0) {
                const auto& sourceNormal = sourceNormals[i];
                const auto& targetNormal = targetNormals[match.idx];

                // Check if normals are valid
                if (!sourceNormal.allFinite() || !targetNormal.allFinite()) {
                    match.idx = -1;
                    prunedCount++;
                    continue;
                }

                // Compute angle between normals (more lenient threshold)
                float dotProduct = sourceNormal.dot(targetNormal);
                float angle = acos(std::min(1.0f, std::max(-1.0f, dotProduct)));
                
                // Use a more lenient threshold (90 degrees instead of 60)
                if (angle > M_PI / 2.0f) {
                    match.idx = -1;
                    prunedCount++;
                }
            }
        }
        
        std::cout << "Pruned " << prunedCount << " correspondences due to normal angle." << std::endl;
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
                problem.AddResidualBlock(PointToPointConstraint::create(sourcePoint, targetPoint, 1.0f), nullptr, poseIncrement.getData());
            


                if (m_bUsePointToPlaneConstraints) {
                    const auto& targetNormal = targetNormals[match.idx];

                    if (!targetNormal.allFinite())
                        continue;

                    // TODO: Create a new point-to-plane cost function and add it as constraint (i.e. residual block) 
                    // to the Ceres problem.
                    problem.AddResidualBlock(PointToPlaneConstraint::create(sourcePoint, targetPoint, targetNormal, 1.0f), nullptr, poseIncrement.getData());



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
    if (nPoints == 0) return Matrix4f::Identity();

    // Build the system A*x = b for the equation (R*s + t - d) · n = 0
    // Linearized for small rotation R ≈ I + [α,β,γ]x:
    // ( (I + [α,β,γ]x)s + t - d ) · n = 0
    // ( s + (α,β,γ) x s + t - d ) · n = 0
    // ( (α,β,γ) x s ) · n + (t) · n = (d - s) · n
    // [ (s x n)^T   n^T ] [α,β,γ,tx,ty,tz]^T = n · (d - s)

    MatrixXf A = MatrixXf::Zero(nPoints, 6);
    VectorXf b = VectorXf::Zero(nPoints);

    for (unsigned i = 0; i < nPoints; i++) {
        const auto& s = sourcePoints[i];
        const auto& d = targetPoints[i];
        // For a linear system, we need the normal of the *target* point from the original (non-transformed) cloud.
        // However, since we don't have the original correspondences, we use the provided targetNormals list.
        // This assumes targetNormals corresponds to targetPoints.
        const auto& n = targetNormals[i];

        // Build the i-th row of matrix A
        Vector<float, 6> A_row;
        A_row.head<3>() = s.cross(n); // Part for rotation
        A_row.tail<3>() = n;          // Part for translation
        A.row(i) = A_row;

        // Build the i-th element of vector b
        b(i) = n.dot(d - s);
    }

    // TODO: Solve the system. Your original code for this was correct.
    VectorXf x = A.colPivHouseholderQr().solve(b);

    // Extract rotation (α, β, γ) and translation (tx, ty, tz)
    float alpha = x(0), beta = x(1), gamma = x(2);
    Vector3f translation = x.tail<3>();

    // Build the pose matrix from the small angle approximations
    Matrix3f rotation;
    rotation = AngleAxisf(gamma, Vector3f::UnitZ()) *
               AngleAxisf(beta, Vector3f::UnitY()) *
               AngleAxisf(alpha, Vector3f::UnitX());

    // TODO: Build the pose matrix. Your original code for this was also correct.
    Matrix4f estimatedPose = Matrix4f::Identity();
    estimatedPose.block<3, 3>(0, 0) = rotation;
    estimatedPose.block<3, 1>(0, 3) = translation;

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
            T(target_point.x()) - transformed_point_T[0],
            T(target_point.y()) - transformed_point_T[1],
            T(target_point.z()) - transformed_point_T[2]
        };

        // The residual is the dot product of the difference with the target normal.
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

            // 1. Transform source points with the current estimated pose to find correspondences.
            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose.cast<float>());
            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose.cast<float>());

            // 2. Find correspondences using the nearest neighbor search.
            std::cout << "Matching points..." << std::endl;
            auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
            pruneCorrespondences(transformedNormals, target.getNormals(), matches);
            std::cout << "Matching complete." << std::endl;

            // 3. Set up and solve the non-linear least squares problem with Ceres.
            ceres::Problem problem;
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
                const Vector3f& target_normal_f = target.getNormals()[matches[j].idx];

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
                    cost_function = new ceres::AutoDiffCostFunction<PointToPlaneError, 1, 6>(
                        new PointToPlaneError(source_pt, target_pt, target_normal)
                    );
                } else {
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
            options.max_num_iterations = 20; // Increased iterations for robustness
            options.minimizer_progress_to_stdout = false; // Set to true for detailed debug info
            
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            
            // 5. Convert the resulting increment to a matrix and pre-multiply it to the current pose.
            Matrix4d increment_matrix = convertIncrementToMatrix(pose_increment);
            estimatedPose = increment_matrix * estimatedPose;
            
            // Output summary for this iteration
            double angle, tx, ty, tz;
            getIncrementAsAngleAndTranslation(pose_increment, angle, tx, ty, tz);
            std::cout << "Pose update - Translation: " << Vector3d(tx, ty, tz).norm() << ", Rotation: " << (angle * 180.0 / M_PI) << " deg" << std::endl;
            
            // Convergence check
            if (Vector3d(tx, ty, tz).norm() < 1e-4 && angle < 1e-4) {
                 std::cout << "Converged after " << i+1 << " iterations." << std::endl;
                 break;
            }
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

class SymmetricPointToPlaneConstraint {
    public:
        SymmetricPointToPlaneConstraint(
            const Vector3f& sourcePoint, const Vector3f& sourceNormal,
            const Vector3f& targetPoint, const Vector3f& targetNormal,
            const float weight) :
            m_sourcePoint{ sourcePoint },
            m_sourceNormal{ sourceNormal },
            m_targetPoint{ targetPoint },
            m_targetNormal{ targetNormal },
            m_weight{ weight }
        { }
    
        template <typename T>
        bool operator()(const T* const pose, T* residuals) const {
            // Create PoseIncrement object to handle the transformation
            PoseIncrement<T> poseIncrement(const_cast<T*>(pose));
    
            // Templated versions of input points and normals
            T sourcePoint_T[3];
            fillVector(m_sourcePoint, sourcePoint_T);
    
            T sourceNormal_T[3];
            fillVector(m_sourceNormal, sourceNormal_T);
    
            T targetPoint_T[3];
            fillVector(m_targetPoint, targetPoint_T);
    
            T targetNormal_T[3];
            fillVector(m_targetNormal, targetNormal_T);
    
            // 1. Transform source point
            T transformedSourcePoint[3];
            poseIncrement.apply(sourcePoint_T, transformedSourcePoint);
    
            // 2. Transform source normal
            T transformedSourceNormal[3];
            poseIncrement.applyNormal(sourceNormal_T, transformedSourceNormal);
            
            // Normalize transformed source normal
            T tsn_magnitude = ceres::sqrt(transformedSourceNormal[0]*transformedSourceNormal[0] + transformedSourceNormal[1]*transformedSourceNormal[1] + transformedSourceNormal[2]*transformedSourceNormal[2]);
            if (tsn_magnitude < T(1e-9)) {
                // Degenerate normal, return zero residual
                residuals[0] = T(0);
                return true;
            }
            T normalizedTransformedSourceNormal[3] = {
                transformedSourceNormal[0] / tsn_magnitude,
                transformedSourceNormal[1] / tsn_magnitude,
                transformedSourceNormal[2] / tsn_magnitude
            };
    
    
            // Normalize target normal
            T tn_magnitude = ceres::sqrt(targetNormal_T[0]*targetNormal_T[0] + targetNormal_T[1]*targetNormal_T[1] + targetNormal_T[2]*targetNormal_T[2]);
            if (tn_magnitude < T(1e-9)) {
                 // Degenerate normal, return zero residual
                residuals[0] = T(0);
                return true;
            }
            T normalizedTargetNormal[3] = {
                targetNormal_T[0] / tn_magnitude,
                targetNormal_T[1] / tn_magnitude,
                targetNormal_T[2] / tn_magnitude
            };
    
            // 3. Calculate difference vector
            T diff[3];
            diff[0] = transformedSourcePoint[0] - targetPoint_T[0];
            diff[1] = transformedSourcePoint[1] - targetPoint_T[1];
            diff[2] = transformedSourcePoint[2] - targetPoint_T[2];
    
            // 4. Calculate combined normal (n_p + n_q)
            T combinedNormal[3];
            combinedNormal[0] = normalizedTransformedSourceNormal[0] + normalizedTargetNormal[0];
            combinedNormal[1] = normalizedTransformedSourceNormal[1] + normalizedTargetNormal[1];
            combinedNormal[2] = normalizedTransformedSourceNormal[2] + normalizedTargetNormal[2];
    
            // 5. Calculate residual: (transformed_source_point - target_point) . (transformed_source_normal + target_normal)
            T dotProduct = diff[0] * combinedNormal[0] +
                           diff[1] * combinedNormal[1] +
                           diff[2] * combinedNormal[2];
    
            // 6. Apply the weight
            T sqrtWeight = T(sqrt(m_weight));
            residuals[0] = sqrtWeight * dotProduct;
    
            return true;
        }
    
        static ceres::CostFunction* create(
            const Vector3f& sourcePoint, const Vector3f& sourceNormal,
            const Vector3f& targetPoint, const Vector3f& targetNormal,
            const float weight) {
            return new ceres::AutoDiffCostFunction<SymmetricPointToPlaneConstraint, 1, 6>(
                new SymmetricPointToPlaneConstraint(sourcePoint, sourceNormal, targetPoint, targetNormal, weight)
            );
        }
    
    protected:
        const Vector3f m_sourcePoint;
        const Vector3f m_sourceNormal;
        const Vector3f m_targetPoint;
        const Vector3f m_targetNormal;
        const float m_weight;
    };
    
    