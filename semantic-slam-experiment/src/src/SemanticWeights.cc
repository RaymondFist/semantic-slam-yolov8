#include "SemanticWeights.h"

namespace semantic_slam {

static const std::unordered_map<int, SemanticClass> COCO_TO_SEMANTIC = {
    // Structural / static objects
    {8,  SemanticClass::BUILDING},        // boat (outdoor structure)
    {10, SemanticClass::TRAFFIC_SIGN},    // fire hydrant
    {11, SemanticClass::TRAFFIC_SIGN},    // stop sign
    {13, SemanticClass::UNKNOWN},         // parking meter
    {14, SemanticClass::UNKNOWN},         // bench

    // Traffic infrastructure
    {9,  SemanticClass::TRAFFIC_LIGHT},   // traffic light

    // Vegetation / natural
    {15, SemanticClass::VEGETATION},      // bird (natural)
    {56, SemanticClass::BUILDING},        // chair (indoor structural)
    {57, SemanticClass::BUILDING},        // couch (indoor structural)
    {58, SemanticClass::UNKNOWN},         // potted plant
    {59, SemanticClass::BUILDING},        // bed
    {60, SemanticClass::BUILDING},        // dining table
    {61, SemanticClass::BUILDING},        // toilet
    {62, SemanticClass::UNKNOWN},         // tv (potentially dynamic screen)
    {63, SemanticClass::BUILDING},        // laptop
    {67, SemanticClass::BUILDING},        // cell phone

    // Road infrastructure
    {6,  SemanticClass::ROAD},            // train (on rails = infrastructure)

    // Dynamic objects — mapped to UNKNOWN so they get unknown_weight
    {0,  SemanticClass::UNKNOWN},         // person
    {1,  SemanticClass::UNKNOWN},         // bicycle
    {2,  SemanticClass::UNKNOWN},         // car
    {3,  SemanticClass::UNKNOWN},         // motorcycle
    {5,  SemanticClass::UNKNOWN},         // bus
    {7,  SemanticClass::UNKNOWN},         // truck
    {16, SemanticClass::UNKNOWN},         // dog
    {17, SemanticClass::UNKNOWN},         // cat
};

const std::unordered_map<int, SemanticClass>& SemanticWeights::getCocoMapping()
{
    return COCO_TO_SEMANTIC;
}

SemanticWeights::SemanticWeights(const Config& config) : _config(config) {}

SemanticClass SemanticWeights::classifyCocoClass(int coco_class_id) {
    auto it = COCO_TO_SEMANTIC.find(coco_class_id);
    if (it != COCO_TO_SEMANTIC.end()) return it->second;
    return SemanticClass::UNKNOWN;
}

double SemanticWeights::getWeight(int coco_class_id) const {
    return getWeightForClass(classifyCocoClass(coco_class_id));
}

double SemanticWeights::getWeightForClass(SemanticClass sc) const {
    switch (sc) {
        case SemanticClass::BUILDING:       return _config.building_weight;
        case SemanticClass::TRAFFIC_SIGN:   return _config.traffic_sign_weight;
        case SemanticClass::TRAFFIC_LIGHT:  return _config.traffic_light_weight;
        case SemanticClass::ROAD:           return _config.road_weight;
        case SemanticClass::VEGETATION:     return _config.vegetation_weight;
        case SemanticClass::UNKNOWN:
        default:                            return _config.unknown_weight;
    }
}

} // namespace semantic_slam