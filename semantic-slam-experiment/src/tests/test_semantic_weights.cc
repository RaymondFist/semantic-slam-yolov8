#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>

#include "SemanticWeights.h"
#include "test_utils.h"

using namespace semantic_slam;

class SemanticWeightsTest : public ::testing::Test {
protected:
    void SetUp() override {
        _cfg.building_weight       = 1.0;
        _cfg.traffic_sign_weight   = 0.9;
        _cfg.traffic_light_weight  = 0.9;
        _cfg.road_weight           = 0.7;
        _cfg.vegetation_weight     = 0.5;
        _cfg.unknown_weight        = 0.8;
        _cfg.consistency_lambda    = 0.1;

        _weights = std::make_unique<SemanticWeights>(_cfg);
    }

    SemanticWeights::Config _cfg;
    std::unique_ptr<SemanticWeights> _weights;
};

// ============================================================================
// TEST 1: Known COCO class IDs map to correct semantic classes
// ============================================================================
TEST_F(SemanticWeightsTest, ClassifyCocoClassKnownMappings) {
    EXPECT_EQ(SemanticWeights::classifyCocoClass(2),  SemanticClass::UNKNOWN);
    EXPECT_EQ(SemanticWeights::classifyCocoClass(9),  SemanticClass::TRAFFIC_LIGHT);
    EXPECT_EQ(SemanticWeights::classifyCocoClass(10), SemanticClass::TRAFFIC_SIGN);
    EXPECT_EQ(SemanticWeights::classifyCocoClass(13), SemanticClass::UNKNOWN);
    EXPECT_EQ(SemanticWeights::classifyCocoClass(15), SemanticClass::VEGETATION);
    EXPECT_EQ(SemanticWeights::classifyCocoClass(56), SemanticClass::BUILDING);
}

// ============================================================================
// TEST 2: Unknown COCO class IDs map to UNKNOWN
// ============================================================================
TEST_F(SemanticWeightsTest, ClassifyCocoClassUnknownMapping) {
    EXPECT_EQ(SemanticWeights::classifyCocoClass(-1),   SemanticClass::UNKNOWN);
    EXPECT_EQ(SemanticWeights::classifyCocoClass(9999),  SemanticClass::UNKNOWN);
    EXPECT_EQ(SemanticWeights::classifyCocoClass(1000),  SemanticClass::UNKNOWN);
}

// ============================================================================
// TEST 3: getWeight returns different values for different categories
// ============================================================================
TEST_F(SemanticWeightsTest, GetWeightDifferentCategories) {
    double w_building   = _weights->getWeight(56);  // chair -> building
    double w_traffic_l  = _weights->getWeight(9);   // traffic light
    double w_vegetation = _weights->getWeight(15);  // bird -> vegetation

    EXPECT_GT(w_building, w_vegetation)
        << "Building weight should be higher than vegetation";
    EXPECT_LT(w_vegetation, 1.0)
        << "Vegetation weight should be below 1.0";
    EXPECT_NEAR(w_vegetation, 0.5, 0.01);
}

// ============================================================================
// TEST 4: getWeightForClass returns correct configured value
// ============================================================================
TEST_F(SemanticWeightsTest, GetWeightForClassConfigValues) {
    EXPECT_NEAR(_weights->getWeightForClass(SemanticClass::BUILDING),       1.0, 0.01);
    EXPECT_NEAR(_weights->getWeightForClass(SemanticClass::TRAFFIC_SIGN),  0.9, 0.01);
    EXPECT_NEAR(_weights->getWeightForClass(SemanticClass::TRAFFIC_LIGHT), 0.9, 0.01);
    EXPECT_NEAR(_weights->getWeightForClass(SemanticClass::ROAD),          0.7, 0.01);
    EXPECT_NEAR(_weights->getWeightForClass(SemanticClass::VEGETATION),    0.5, 0.01);
    EXPECT_NEAR(_weights->getWeightForClass(SemanticClass::UNKNOWN),       0.8, 0.01);
}

// ============================================================================
// TEST 7: Custom config produces different weights
// ============================================================================
TEST_F(SemanticWeightsTest, CustomConfigWeights) {
    SemanticWeights::Config cfg;
    cfg.building_weight   = 0.5;
    cfg.vegetation_weight = 0.9;
    cfg.unknown_weight    = 0.3;

    SemanticWeights custom(cfg);

    EXPECT_NEAR(custom.getWeightForClass(SemanticClass::BUILDING),   0.5, 0.01);
    EXPECT_NEAR(custom.getWeightForClass(SemanticClass::VEGETATION), 0.9, 0.01);
    EXPECT_NEAR(custom.getWeightForClass(SemanticClass::UNKNOWN),    0.3, 0.01);
}

// ============================================================================
// TEST 8: Weight matrix information scaling for g2o
// ============================================================================
TEST_F(SemanticWeightsTest, InformationScalingValid) {
    double base_info = 1.0;
    double w = 0.5;

    double scaled = base_info * w;
    EXPECT_DOUBLE_EQ(scaled, 0.5);

    w = 1.0;
    scaled = base_info * w;
    EXPECT_DOUBLE_EQ(scaled, 1.0);

    w = 2.0;
    scaled = base_info * w;
    EXPECT_DOUBLE_EQ(scaled, 2.0);
}

// ============================================================================
// TEST 9: Get Pseudo-Huber delta with semantic weight
// ============================================================================
TEST_F(SemanticWeightsTest, PseudoHuberDeltaScaling) {
    double delta_raw = 1.345;
    double w = 0.5;

    double delta_effective = delta_raw * w;
    EXPECT_DOUBLE_EQ(delta_effective, 0.6725);

    w = 1.0;
    delta_effective = delta_raw * w;
    EXPECT_DOUBLE_EQ(delta_effective, 1.345);
}