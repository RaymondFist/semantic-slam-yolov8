#ifndef ORB_SLAM3_MAPPOINT_EXT_H
#define ORB_SLAM3_MAPPOINT_EXT_H

// ============================================================================
// PATCH FILE: ORB-SLAM3 Integration Modifications
// ============================================================================
// Target: ORB-SLAM3 commit 67a3831
// Files to modify: MapPoint.h, Frame.h, Tracking.cc, Optimizer.cc,
//                  LocalMapping.cc, System.h, System.cc
// ============================================================================

// ============================================================================
// 1. MapPoint.h — Add semantic class ID
// ============================================================================
//
// In class MapPoint, add the following member and accessor after line ~110
// (near nObs, mnFirstKFid, etc.):
//
//   // ---- Semantic Extension ----
//   public:
//       void SetSemanticClass(int class_id) { mSemanticClass = class_id; }
//       int  GetSemanticClass() const       { return mSemanticClass; }
//       bool HasSemanticClass() const       { return mSemanticClass >= 0; }
//
//       void SetSemanticWeight(double w)    { mSemanticWeight = w; }
//       double GetSemanticWeight() const     { return mSemanticWeight; }
//
//   protected:
//       int    mSemanticClass  = -1;   // COCO class ID, -1 = unknown
//       double mSemanticWeight = 0.8;  // default unknown weight

// ============================================================================
// 2. Frame.h — Add dynamic feature mask
// ============================================================================
//
// In class Frame, add after line ~200 (near mvbOutlier):
//
//   // ---- Semantic Extension ----
//   public:
//       bool IsDynamicFeature(size_t idx) const {
//           if (idx >= mvDynamicMask.size()) return false;
//           return mvDynamicMask[idx];
//       }
//       size_t CountDynamicFeatures() const {
//           return std::count(mvDynamicMask.begin(), mvDynamicMask.end(), true);
//       }
//       void SetDynamicMask(const std::vector<bool>& mask) {
//           mvDynamicMask = mask;
//       }
//       const std::vector<bool>& GetDynamicMask() const {
//           return mvDynamicMask;
//       }
//
//   protected:
//       std::vector<bool> mvDynamicMask;

// ============================================================================
// 3. Tracking.h — Add semantic system reference
// ============================================================================
//
// Add forward declaration near top:
//   namespace semantic_slam { class SemanticSLAM; }
//
// Add member to Tracking class:
//   semantic_slam::SemanticSLAM* mpSemanticSLAM = nullptr;
//
// Add to constructor parameter list:
//   semantic_slam::SemanticSLAM* pSemanticSLAM = nullptr
//
// Add to constructor body:
//   mpSemanticSLAM = pSemanticSLAM;

#endif // ORB_SLAM3_MAPPOINT_EXT_H