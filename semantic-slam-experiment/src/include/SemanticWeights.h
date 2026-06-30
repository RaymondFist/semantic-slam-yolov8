#ifndef SEMANTICWEIGHTS_H
#define SEMANTICWEIGHTS_H

#include <unordered_map>
#include <string>
#include <vector>
#include <Eigen/Core>

namespace semantic_slam {

enum class SemanticClass : int {
    BUILDING       = 0,
    TRAFFIC_SIGN   = 1,
    TRAFFIC_LIGHT  = 2,
    ROAD           = 3,
    VEGETATION     = 4,
    UNKNOWN        = 99
};

class SemanticWeights {
public:
    struct Config {
        double building_weight      = 1.0;
        double traffic_sign_weight  = 0.9;
        double traffic_light_weight = 0.9;
        double road_weight          = 0.7;
        double vegetation_weight    = 0.5;
        double unknown_weight       = 0.8;

        double consistency_lambda   = 0.1;   // semantic term weight in total cost
    };

    SemanticWeights(const Config& config);

    static SemanticClass classifyCocoClass(int coco_class_id);

    double getWeight(int coco_class_id) const;
    double getWeightForClass(SemanticClass sc) const;

    const Config& getConfig() const { return _config; }

private:
    Config _config;

    static const std::unordered_map<int, SemanticClass>& getCocoMapping();
};

} // namespace semantic_slam

#endif // SEMANTICWEIGHTS_H