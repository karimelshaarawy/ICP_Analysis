#pragma once

/*
 * ICP Optimizer Configuration:
 * 
 * Choose one of the following ICP implementations:
 * - USE_HIERARCHICAL_ICP = 1: Uses HierarchicalICPOptimizer (3-level multi-resolution, fastest)
 * - USE_LINEAR_ICP = 1: Uses LinearICPOptimizer (fast, works well for small rotations)
 * - USE_LM_ICP = 1: Uses LevenbergMarquardtICPOptimizer (robust, handles large rotations)
 * - USE_SYMMETRIC_ICP = 1: Uses AlternatingSymmetricICPOptimizer (symmetric approach)
 * - All = 0: Uses CeresICPOptimizer (default, most robust but requires Ceres)
 * 
 * Constraint Type:
 * - USE_POINT_TO_PLANE = 1: Uses point-to-plane constraints (usually better for surfaces)
 * - USE_POINT_TO_PLANE = 0: Uses point-to-point constraints (faster, works well for point clouds)
 * 
 * Colored ICP:
 * - USE_COLORED_ICP = 1: Uses color information in addition to geometry for better alignment
 * - USE_COLORED_ICP = 0: Uses only geometric information (faster)
 * 
 * Downsampling Level:
 * - DOWNSAMPLING_LEVEL = "low": No downsampling (highest quality, slowest)
 * - DOWNSAMPLING_LEVEL = "medium": 4x downsampling (balanced performance)
 * - DOWNSAMPLING_LEVEL = "high": 8x downsampling (fastest processing)
 * 
 * Test Scenario:
 * - RUN_SHAPE_ICP = 1: Run bunny alignment test (faster, good for development)
 * - RUN_SEQUENCE_ICP = 1: Run room reconstruction test (comprehensive, slower)
 * 
 * Examples:
 * - For bunny alignment with LM-ICP and point-to-plane: USE_LM_ICP=1, USE_POINT_TO_PLANE=1
 * - For room reconstruction with colored symmetric ICP: USE_SYMMETRIC_ICP=1, USE_POINT_TO_PLANE=1, USE_COLORED_ICP=1
 * - For hierarchical ICP with colored constraints: USE_HIERARCHICAL_ICP=1, USE_COLORED_ICP=1
 * - For fast processing: DOWNSAMPLING_LEVEL="high"
 * - For high accuracy: DOWNSAMPLING_LEVEL="low"
 */

#define SHOW_BUNNY_CORRESPONDENCES 0

// ICP Optimizer Selection (only one should be 1)
#define USE_POINT_TO_PLANE 1
#define USE_LINEAR_ICP 0
#define USE_LM_ICP 0
#define USE_SYMMETRIC_ICP 1
#define USE_HIERARCHICAL_ICP 0
#define USE_COLORED_ICP 0

// Test Scenario Selection
#define RUN_SHAPE_ICP 0
#define RUN_SEQUENCE_ICP 1

// Downsampling configuration
// Choose one of: "low", "medium", "high"
// - low: downsampleFactor = 1 (no downsampling, highest quality, slowest)
// - medium: downsampleFactor = 4 (75% reduction, balanced performance)
// - high: downsampleFactor = 8 (87.5% reduction, fastest processing)
#define DOWNSAMPLING_LEVEL "medium"
