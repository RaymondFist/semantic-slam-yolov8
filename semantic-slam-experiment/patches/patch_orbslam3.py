#!/usr/bin/env python3
"""
Patch ORB-SLAM3 to integrate SemanticSLAM dynamic feature filtering.

Usage:
    python3 patch_orbslam3.py /root/ORB_SLAM3 [~/autodl-tmp/Script/semantic-slam-yolov8/src]

This script modifies the following ORB-SLAM3 source files:
  - include/Frame.h       : Add dynamic feature mask + accessors
  - src/Frame.cc          : Add mvDynamicMask to copy constructor initializer list
  - include/MapPoint.h    : Add semantic class and weight
  - include/Tracking.h    : Add SemanticSLAM pointer + mLastFrameImage
  - include/System.h      : Add SemanticSLAM member
  - src/Tracking.cc       : Inject semantic detection + filtering before Track()
  - src/System.cc         : Initialize/shutdown SemanticSLAM
  - src/Optimizer.cc      : Use semantic weights in LocalBA
  - src/ORBmatcher.cc     : Skip dynamic features in matching
  - CMakeLists.txt        : Link SemanticSLAM library

Original files are backed up with .orig suffix.
"""

import os
import sys
import shutil
import re

# ============================================================================
# Configuration
# ============================================================================
ORB_SLAM3_ROOT = ""
SEMANTIC_SLAM_ROOT = ""

def backup_file(filepath):
    backup = filepath + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        print(f"  [BACKUP] {filepath} -> {backup}")

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

def write_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [PATCHED] {filepath}")

def is_patched(content, marker):
    return marker in content


# ============================================================================
# Patch 1: Frame.h — Add dynamic feature mask
# ============================================================================
def patch_frame_h():
    filepath = os.path.join(ORB_SLAM3_ROOT, "include", "Frame.h")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_FRAME"):
        print(f"  [SKIP] {filepath} already patched")
        return

    backup_file(filepath)

    # 1a. Add mvDynamicMask member after mvbOutlier
    # ORB-SLAM3 Frame.h actual structure (from GitHub):
    #   std::vector<bool> mvbOutlier;
    #   int mnCloseMPs;
    # Note: 4-space indent for member declarations
    content = content.replace(
        "std::vector<bool> mvbOutlier;\n"
        "    int mnCloseMPs;",
        "std::vector<bool> mvbOutlier;\n"
        "    // SEMANTIC_SLAM_PATCH_FRAME: Dynamic feature mask\n"
        "    std::vector<bool> mvDynamicMask;\n"
        "    int mnCloseMPs;"
    )

    # 1b. Add accessor methods before the closing "private:" section
    # ORB-SLAM3 Frame.h uses "private:" for pose members
    accessor_code = (
        "\n"
        "    // SEMANTIC_SLAM_PATCH_FRAME: Dynamic feature accessors\n"
        "    void SetDynamicMask(const std::vector<bool>& mask) { mvDynamicMask = mask; }\n"
        "    bool IsDynamicFeature(size_t idx) const {\n"
        "        if (idx >= mvDynamicMask.size()) return false;\n"
        "        return mvDynamicMask[idx];\n"
        "    }\n"
        "    size_t CountDynamicFeatures() const {\n"
        "        return std::count(mvDynamicMask.begin(), mvDynamicMask.end(), true);\n"
        "    }\n"
        "    const std::vector<bool>& GetDynamicMask() const { return mvDynamicMask; }\n"
        "\n"
    )
    # Insert before the FIRST "private:" in the file
    if "private:" in content:
        content = content.replace("private:", accessor_code + "private:", 1)
    else:
        # Fallback: insert before the closing of the class
        content = content.replace("};\n}// namespace ORB_SLAM",
                                  accessor_code + "};\n}// namespace ORB_SLAM")

    write_file(filepath, content)


# ============================================================================
# Patch 1b: Frame.cc — Add mvDynamicMask to copy constructor initializer list
# ============================================================================
def patch_frame_cc():
    filepath = os.path.join(ORB_SLAM3_ROOT, "src", "Frame.cc")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_FRAME_CC"):
        print(f"  [SKIP] {filepath} already patched")
        return

    backup_file(filepath)

    # ORB-SLAM3 Frame copy constructor initializer list contains mvbOutlier.
    # We add mvDynamicMask initialization right after mvbOutlier in the
    # copy constructor's initializer list.
    # Typical pattern: "mvbOutlier(F.mvbOutlier),"
    content = content.replace(
        "mvbOutlier(F.mvbOutlier),",
        "mvbOutlier(F.mvbOutlier),\n"
        "    mvDynamicMask(F.mvDynamicMask),  // SEMANTIC_SLAM_PATCH_FRAME_CC"
    )

    write_file(filepath, content)


# ============================================================================
# Patch 2: MapPoint.h — Add semantic class and weight
# ============================================================================
def patch_mappoint_h():
    filepath = os.path.join(ORB_SLAM3_ROOT, "include", "MapPoint.h")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_MAPPOINT"):
        print(f"  [SKIP] {filepath} already patched")
        return

    backup_file(filepath)

    # Add semantic members after nObs — keep them in the same access section
    # (nObs is public, so the new members stay public too)
    content = content.replace(
        "int nObs;",
        "int nObs;\n"
        "\n"
        "        // SEMANTIC_SLAM_PATCH_MAPPOINT: Semantic extension\n"
        "        void SetSemanticClass(int class_id) { mSemanticClass = class_id; }\n"
        "        int  GetSemanticClass() const       { return mSemanticClass; }\n"
        "        bool HasSemanticClass() const       { return mSemanticClass >= 0; }\n"
        "\n"
        "        void SetSemanticWeight(double w)    { mSemanticWeight = w; }\n"
        "        double GetSemanticWeight() const     { return mSemanticWeight; }\n"
        "\n"
        "        int    mSemanticClass  = -1;   // COCO class ID, -1 = unknown\n"
        "        double mSemanticWeight = 0.8;  // default unknown weight"
    )

    write_file(filepath, content)


# ============================================================================
# Patch 3: Tracking.h — Add SemanticSLAM pointer
# ============================================================================
def patch_tracking_h():
    filepath = os.path.join(ORB_SLAM3_ROOT, "include", "Tracking.h")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_TRACKING"):
        print(f"  [SKIP] {filepath} already patched")
        return

    backup_file(filepath)

    # Add forward declaration BEFORE namespace ORB_SLAM3 (guarantees global namespace)
    fwd_decl = (
        "\n"
        "// SEMANTIC_SLAM_PATCH_TRACKING: Forward declaration\n"
        "namespace semantic_slam { class SemanticSLAM; }\n"
    )
    if "namespace ORB_SLAM3" in content:
        content = content.replace(
            "namespace ORB_SLAM3",
            fwd_decl + "namespace ORB_SLAM3",
            1
        )
    else:
        # Fallback: add before class definition
        content = content.replace(
            "class Tracking",
            fwd_decl + "\nclass Tracking",
            1
        )

    # Add member variables before the first "protected:" section
    # Tracking.h has: protected: after the public section
    # We add mpSemanticSLAM and mLastFrameImage as protected members
    member_code = (
        "\n"
        "    // SEMANTIC_SLAM_PATCH_TRACKING: Semantic SLAM module\n"
        "    ::semantic_slam::SemanticSLAM* mpSemanticSLAM = nullptr;\n"
        "    cv::Mat mLastFrameImage;  // Store for optical flow\n"
        "\n"
    )
    if "protected:" in content:
        content = content.replace("protected:", member_code + "protected:", 1)
    else:
        # Fallback: add before closing of class
        content = content.replace(
            "};\n} //namespace ORB_SLAM",
            member_code + "};\n} //namespace ORB_SLAM"
        )

    write_file(filepath, content)


# ============================================================================
# Patch 4: System.h — Add SemanticSLAM member
# ============================================================================
def patch_system_h():
    filepath = os.path.join(ORB_SLAM3_ROOT, "include", "System.h")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_SYSTEM"):
        print(f"  [SKIP] {filepath} already patched")
        return

    backup_file(filepath)

    # Add forward declaration BEFORE namespace ORB_SLAM3 (guarantees global namespace)
    fwd_decl = (
        "\n"
        "// SEMANTIC_SLAM_PATCH_SYSTEM: Semantic SLAM integration\n"
        "namespace semantic_slam { class SemanticSLAM; }\n"
    )
    if "namespace ORB_SLAM3" in content:
        content = content.replace(
            "namespace ORB_SLAM3",
            fwd_decl + "namespace ORB_SLAM3",
            1
        )

    # Add mpSemanticSLAM member in the private section (after settings_)
    # ORB-SLAM3 System.h private section ends with: "Settings* settings_;\n};"
    content = content.replace(
        "Settings* settings_;",
        "Settings* settings_;\n"
        "\n"
        "        // SEMANTIC_SLAM_PATCH_SYSTEM: Semantic SLAM module\n"
        "        ::semantic_slam::SemanticSLAM* mpSemanticSLAM = nullptr;"
    )

    # Add public accessor — insert before the first "private:"
    accessor = (
        "\n"
        "    // SEMANTIC_SLAM_PATCH_SYSTEM: Accessor\n"
        "    ::semantic_slam::SemanticSLAM* GetSemanticSLAM() { return mpSemanticSLAM; }\n"
        "\n"
    )
    if "private:" in content:
        content = content.replace("private:", accessor + "private:", 1)

    write_file(filepath, content)


# ============================================================================
# Patch 5: Tracking.cc — Inject semantic detection + filtering
#
# CRITICAL FIX: The old script used sequential Track();\n replacement which
# broke after the first replacement. Now we use regex to find each GrabImage*
# function body and inject the semantic filtering block before its Track() call.
# ============================================================================
def patch_tracking_cc():
    filepath = os.path.join(ORB_SLAM3_ROOT, "src", "Tracking.cc")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_TRACKING_CC"):
        print(f"  [SKIP] {filepath} already patched")
        return

    backup_file(filepath)

    # 5a. Add includes at the top
    content = content.replace(
        '#include "Tracking.h"',
        '#include "Tracking.h"\n'
        '\n'
        '// SEMANTIC_SLAM_PATCH_TRACKING_CC: Semantic SLAM integration\n'
        '#include <SemanticSLAM.h>\n'
        '#include <DynamicFeatureFilter.h>\n'
        '#include <opencv2/video/tracking.hpp>\n'
    )

    # 5b. Build the semantic filtering code block (shared by all 3 GrabImage* functions)
    # NOTE: ORB-SLAM3 uses mImGray (member variable) not imGray
    # NOTE: Dynamic features are stored in mvDynamicMask via SetDynamicMask().
    #       They are filtered in SearchByProjection/SearchByBoW via IsDynamicFeature().
    #       Do NOT set mvbOutlier directly — it causes g2o double-free in PoseOptimization.
    # NOTE: Added mImGray.empty() guard to prevent crashes when Track() is called
    #       in error states (e.g., NO_IMAGES_YET).
    # NOTE: mCurrentFrame is a Frame object (not a pointer), so .mvKeysUn is safe.
    semantic_filter_block = (
        '    // SEMANTIC_SLAM_PATCH_TRACKING_CC: Semantic filtering\n'
        '    if(mpSemanticSLAM && !mImGray.empty()) {\n'
        '        fprintf(stderr, "[DEBUG_TRACK] Frame %lld: submitFrame+getDetectionResult\\n", (long long)mCurrentFrame.mnId);\n'
        '        mpSemanticSLAM->submitFrame(mImGray, mCurrentFrame.mTimeStamp, mCurrentFrame.mnId);\n'
        '        auto det = mpSemanticSLAM->getDetectionResult(mCurrentFrame.mnId, 50.0);\n'
        '        fprintf(stderr, "[DEBUG_TRACK] Frame %lld: det.valid=%d instances=%zu lastFrame.empty=%d\\n",\n'
        '                (long long)mCurrentFrame.mnId, (int)det.valid, det.instances.size(), (int)mLastFrameImage.empty());\n'
        '        if(det.valid && !det.instances.empty()\n'
        '           && !mLastFrameImage.empty()\n'
        '           && !mCurrentFrame.mvKeysUn.empty()\n'
        '           && mLastFrameImage.size() == mImGray.size()\n'
        '           && mLastFrameImage.type() == mImGray.type()) {\n'
        '            fprintf(stderr, "[DEBUG_TRACK] Frame %lld: entering filter block\\n", (long long)mCurrentFrame.mnId);\n'
        '            // SEMANTIC_SLAM_FIX: try-catch to prevent segfault from\n'
        '            // filterDynamicFeatures (e.g. invalid detection data, OpenCV\n'
        '            // optical flow failure). On exception, skip filtering and\n'
        '            // continue with normal tracking.\n'
        '            try {\n'
        '                fprintf(stderr, "[DEBUG_TRACK] Frame %lld: calling filterDynamicFeatures\\n", (long long)mCurrentFrame.mnId);\n'
        '                // Pass empty prev/curr points so processFrameMasked computes\n'
        '                // optical flow internally (backward flow: curr -> prev)\n'
        '                std::vector<cv::Point2f> no_prev_pts, no_curr_pts;\n'
        '                auto dynamic_mask = mpSemanticSLAM->filterDynamicFeatures(\n'
        '                    mImGray, mLastFrameImage, det,\n'
        '                    mCurrentFrame.mvKeysUn, no_prev_pts, no_curr_pts);\n'
        '                fprintf(stderr, "[DEBUG_TRACK] Frame %lld: filterDynamicFeatures returned size=%zu\\n",\n'
        '                        (long long)mCurrentFrame.mnId, dynamic_mask.size());\n'
        '                mCurrentFrame.SetDynamicMask(dynamic_mask);\n'
        '                fprintf(stderr, "[DEBUG_TRACK] Frame %lld: SetDynamicMask done\\n", (long long)mCurrentFrame.mnId);\n'
        '                // NOTE: Do NOT mark dynamic features as mvbOutlier.\n'
        '                // mvbOutlier is used by PoseOptimization which relies on\n'
        '                // a valid g2o internal state. Marking too many outliers\n'
        '                // (e.g. 644/1012 features) causes LinearSolverDense to\n'
        '                // double-free in destructor (SIGSEGV).\n'
        '                // Dynamic features are already filtered by SearchByProjection\n'
        '                // and SearchByBoW via IsDynamicFeature(mvDynamicMask).\n'
        '            } catch(const std::exception& e) {\n'
        '                std::cerr << "[SemanticSLAM] filterDynamicFeatures exception: "\n'
        '                          << e.what() << " — skipping dynamic filtering" << std::endl;\n'
        '                // CRITICAL: Initialize empty mask on exception to prevent\n'
        '                // mvDynamicMask from being empty (causes g2o crash)\n'
        '                std::vector<bool> empty_mask(mCurrentFrame.mvKeysUn.size(), false);\n'
        '                mCurrentFrame.SetDynamicMask(empty_mask);\n'
        '            } catch(...) {\n'
        '                std::cerr << "[SemanticSLAM] filterDynamicFeatures unknown exception"\n'
        '                          << " — skipping dynamic filtering" << std::endl;\n'
        '                std::vector<bool> empty_mask(mCurrentFrame.mvKeysUn.size(), false);\n'
        '                mCurrentFrame.SetDynamicMask(empty_mask);\n'
        '            }\n'
        '        } else {\n'
        '            fprintf(stderr, "[DEBUG_TRACK] Frame %lld: SKIPPED filter block, initializing empty mask\\n", (long long)mCurrentFrame.mnId);\n'
        '            // CRITICAL: Always initialize mvDynamicMask, even when no detections exist.\n'
        '            // An empty mvDynamicMask causes g2o LinearSolverDense double-free\n'
        '            // in PoseOptimization (SIGSEGV). The mask must always be sized to\n'
        '            // match mvKeysUn.\n'
        '            std::vector<bool> empty_mask(mCurrentFrame.mvKeysUn.size(), false);\n'
        '            mCurrentFrame.SetDynamicMask(empty_mask);\n'
        '        }\n'
        '        mLastFrameImage = mImGray.clone();\n'
        '        fprintf(stderr, "[DEBUG_TRACK] Frame %lld: semantic filter done\\n", (long long)mCurrentFrame.mnId);\n'
        '    }\n'
    )

    # 5b2. After Track(), set semantic class on MapPoints for non-dynamic features.
    # This is needed so that Optimizer.cc can apply semantic weights in LocalBA.
    # Without this, MapPoint.mSemanticClass stays -1 and weights are never used.
    semantic_mappoint_block = (
        '    // SEMANTIC_SLAM_PATCH_TRACKING_CC: Set semantic class on MapPoints\n'
        '    if(mpSemanticSLAM && mCurrentFrame.GetDynamicMask().size() == mCurrentFrame.mvpMapPoints.size()) {\n'
        '        try {\n'
        '            const auto& dmask = mCurrentFrame.GetDynamicMask();\n'
        '            auto det = mpSemanticSLAM->getDetectionResult(mCurrentFrame.mnId, 0.0);\n'
        '            if(det.valid) {\n'
        '                for(size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); ++i) {\n'
        '                    if(mCurrentFrame.mvpMapPoints[i] && !dmask[i] && i < mCurrentFrame.mvKeysUn.size()) {\n'
        '                        const auto& kp = mCurrentFrame.mvKeysUn[i];\n'
        '                        for(const auto& inst : det.instances) {\n'
        '                            if(inst.bbox.contains(kp.pt)) {\n'
        '                                mCurrentFrame.mvpMapPoints[i]->SetSemanticClass(inst.class_id);\n'
        '                                double w = mpSemanticSLAM->getMapPointWeight(inst.class_id);\n'
        '                                mCurrentFrame.mvpMapPoints[i]->SetSemanticWeight(w);\n'
        '                                break;\n'
        '                            }\n'
        '                        }\n'
        '                    }\n'
        '                }\n'
        '            }\n'
        '        } catch(const std::exception& e) {\n'
        '            std::cerr << "[SemanticSLAM] mappoint assignment exception: "\n'
        '                      << e.what() << " — skipping" << std::endl;\n'
        '        } catch(...) {\n'
        '            std::cerr << "[SemanticSLAM] mappoint assignment unknown exception"\n'
        '                      << " — skipping" << std::endl;\n'
        '        }\n'
        '    }\n'
    )

    # 5c. Patch Track() function.
    # Instead of trying to match each GrabImage* function signature (which is
    # fragile across different ORB-SLAM3 versions), we inject the semantic
    # filtering code directly into Track(). This is the common entry point:
    # all GrabImage* functions call Track() after setting up mImGray and
    # mCurrentFrame.
    #
    # We inject:
    #   - semantic_filter_block at the beginning of Track() (after opening brace)
    #   - semantic_mappoint_block at the end of Track() (before closing brace)
    #
    # This is more robust than the previous approach of matching GrabImage*
    # function signatures, which failed with ORB-SLAM3 v1.0.

    # Find "void Tracking::Track()" — the function definition in Tracking.cc
    track_pattern = re.compile(r'void\s+Tracking::Track\s*\(\s*\)')
    m = track_pattern.search(content)
    if m:
        # Find the opening brace after the function signature
        brace_pos = content.find('{', m.end())
        if brace_pos >= 0:
            # Find the matching closing brace
            depth = 0
            close_pos = -1
            for i in range(brace_pos, len(content)):
                if content[i] == '{':
                    depth += 1
                elif content[i] == '}':
                    depth -= 1
                    if depth == 0:
                        close_pos = i
                        break

            if close_pos >= 0:
                # Inject filter block after opening brace
                # Find the first newline after opening brace to get proper indentation
                nl = content.find('\n', brace_pos)
                if nl >= 0 and nl < close_pos:
                    insert_pos = nl + 1
                else:
                    insert_pos = brace_pos + 1

                content = content[:insert_pos] + '\n' + semantic_filter_block + '\n' + content[insert_pos:]

                # Recalculate close_pos (shifted by the injected code)
                # The injected code has len(semantic_filter_block) + 2 newline chars
                shift = len(semantic_filter_block) + 2
                close_pos += shift

                # Inject mappoint block before closing brace
                # Find the last newline before closing brace for proper indentation
                last_nl = content.rfind('\n', brace_pos, close_pos)
                if last_nl >= 0 and last_nl > brace_pos:
                    # Insert after the last statement before closing brace
                    content = content[:close_pos] + '\n' + semantic_mappoint_block + '\n' + content[close_pos:]
                else:
                    content = content[:close_pos] + semantic_mappoint_block + content[close_pos:]

                print("  [OK] Patched Track() with semantic filter + mappoint blocks")
            else:
                print("  [WARN] Could not find closing brace of Track()")
        else:
            print("  [WARN] Could not find opening brace of Track()")
    else:
        print("  [WARN] Track() function not found in Tracking.cc")

    write_file(filepath, content)


# ============================================================================
# Patch 6: System.cc — Initialize/shutdown SemanticSLAM
#
# CRITICAL FIXES:
# - Old script inserted init code BEFORE mpTracker creation, but referenced
#   mpTracker->mpSemanticSLAM. Now init is inserted AFTER mpTracker creation.
# - Old script added an extra "{" after Shutdown() declaration. Fixed.
# ============================================================================
def patch_system_cc():
    filepath = os.path.join(ORB_SLAM3_ROOT, "src", "System.cc")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_SYSTEM_CC"):
        print(f"  [SKIP] {filepath} already patched")
        return

    backup_file(filepath)

    # 6a. Add includes
    content = content.replace(
        '#include "System.h"',
        '#include "System.h"\n'
        '\n'
        '// SEMANTIC_SLAM_PATCH_SYSTEM_CC: Semantic SLAM integration\n'
        '#include <SemanticSLAM.h>\n'
        '#include <YoloDetector.h>\n'
        '#include <DynamicFeatureFilter.h>\n'
        '#include <SemanticWeights.h>\n'
    )

    # 6b. Insert SemanticSLAM initialization AFTER mpTracker creation.
    # Find the line that creates mpTracker and insert after it.
    # The typical ORB-SLAM3 code is:
    #   mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
    #                            mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);
    #
    # We find this multi-line statement by looking for the start pattern
    # and then finding the closing semicolon.

    init_code = (
        '\n'
        '    // SEMANTIC_SLAM_PATCH_SYSTEM_CC: Initialize SemanticSLAM\n'
        '    {\n'
        '        ::semantic_slam::YoloDetector::Config yolo_cfg;\n'
        '        yolo_cfg.conf_threshold = 0.45f;\n'
        '        yolo_cfg.nms_threshold = 0.45f;\n'
        '        yolo_cfg.input_width = 640;\n'
        '        yolo_cfg.input_height = 640;\n'
        '\n'
        '        ::semantic_slam::DynamicFeatureFilter::Config filter_cfg;\n'
        '        filter_cfg.flow_threshold_px = 2.5f;\n'
        '        filter_cfg.lk_window_size = 21;\n'
        '        filter_cfg.lk_max_level = 3;\n'
        '        filter_cfg.min_corners_per_region = 10;\n'
        '        filter_cfg.mask_dilation_kernel = 5;\n'
        '\n'
        '        ::semantic_slam::SemanticWeights::Config weight_cfg;\n'
        '\n'
        '        // Read all semantic config from YAML settings\n'
        '        bool has_yolo_config = false;\n'
        '        {\n'
        '            cv::FileStorage fs(strSettingsFile, cv::FileStorage::READ);\n'
        '            if(fs.isOpened()) {\n'
        '                // NOTE: OpenCV %YAML:1.0 does NOT support nested mappings.\n'
        '                // All keys are flat dot-separated (e.g. yolo_detector.onnx_path)\n'
        '                // Check for yolo_detector.onnx_path as the sentinel key\n'
        '                if(!fs["yolo_detector.onnx_path"].empty()) {\n'
        '                    has_yolo_config = true;\n'
        '                    std::string dd = (std::string)fs["yolo_detector.detection_dir"];\n'
        '                    if(!dd.empty()) yolo_cfg.detection_dir = dd;\n'
        '                    std::string onnx = (std::string)fs["yolo_detector.onnx_path"];\n'
        '                    if(!onnx.empty()) yolo_cfg.onnx_path = onnx;\n'
        '                    std::string names = (std::string)fs["yolo_detector.class_names_path"];\n'
        '                    if(!names.empty()) yolo_cfg.class_names_path = names;\n'
        '                }\n'
        '                if(!fs["dynamic_filter.flow_threshold_px"].empty()) filter_cfg.flow_threshold_px = (float)fs["dynamic_filter.flow_threshold_px"];\n'
        '                if(!fs["dynamic_filter.lk_window_size"].empty())    filter_cfg.lk_window_size = (int)fs["dynamic_filter.lk_window_size"];\n'
        '                if(!fs["dynamic_filter.lk_max_level"].empty())      filter_cfg.lk_max_level = (int)fs["dynamic_filter.lk_max_level"];\n'
        '                if(!fs["dynamic_filter.min_corners_per_region"].empty()) filter_cfg.min_corners_per_region = (int)fs["dynamic_filter.min_corners_per_region"];\n'
        '                if(!fs["dynamic_filter.mask_dilation_kernel"].empty()) filter_cfg.mask_dilation_kernel = (int)fs["dynamic_filter.mask_dilation_kernel"];\n'
        '                if(!fs["semantic_weights.building_weight"].empty())      weight_cfg.building_weight = (double)fs["semantic_weights.building_weight"];\n'
        '                if(!fs["semantic_weights.traffic_sign_weight"].empty())  weight_cfg.traffic_sign_weight = (double)fs["semantic_weights.traffic_sign_weight"];\n'
        '                if(!fs["semantic_weights.road_weight"].empty())          weight_cfg.road_weight = (double)fs["semantic_weights.road_weight"];\n'
        '                if(!fs["semantic_weights.vegetation_weight"].empty())    weight_cfg.vegetation_weight = (double)fs["semantic_weights.vegetation_weight"];\n'
        '                if(!fs["semantic_weights.unknown_weight"].empty())       weight_cfg.unknown_weight = (double)fs["semantic_weights.unknown_weight"];\n'
        '                if(!fs["semantic_weights.consistency_lambda"].empty())   weight_cfg.consistency_lambda = (double)fs["semantic_weights.consistency_lambda"];\n'
        '                fs.release();\n'
        '            }\n'
        '        }\n'
        '\n'
        '        if(!has_yolo_config) {\n'
        '            // No yolo_detector section in YAML → pure ORB-SLAM3 baseline\n'
        '            mpSemanticSLAM = nullptr;\n'
        '            std::cout << "[SemanticSLAM] No yolo_detector config, running baseline." << std::endl;\n'
        '        } else {\n'
        '            mpSemanticSLAM = new ::semantic_slam::SemanticSLAM(\n'
        '                yolo_cfg, filter_cfg, weight_cfg);\n'
        '            if(!mpSemanticSLAM->initialize()) {\n'
        '                std::cerr << "[SemanticSLAM] Failed to initialize semantic module" << std::endl;\n'
        '                delete mpSemanticSLAM;\n'
        '                mpSemanticSLAM = nullptr;\n'
        '            } else {\n'
        '                mpSemanticSLAM->start();\n'
        '                std::cout << "[SemanticSLAM] Initialized successfully" << std::endl;\n'
        '            }\n'
        '        }\n'
        '\n'
        '        // Pass to Tracker\n'
        '        mpTracker->mpSemanticSLAM = mpSemanticSLAM;\n'
        '    }\n'
    )

    # Find "mpTracker = new Tracking(this," and then find the closing ";"
    # The constructor call spans multiple lines in ORB-SLAM3
    tracker_pattern = re.compile(r'mpTracker\s*=\s*new\s+Tracking\s*\(')
    m = tracker_pattern.search(content)
    if m:
        # Find the closing semicolon of this statement
        pos = m.start()
        semi_pos = content.find(';', pos)
        if semi_pos >= 0:
            # Insert after the semicolon + newline
            insert_pos = semi_pos + 1  # after ';'
            # Skip to end of line
            nl_pos = content.find('\n', insert_pos)
            if nl_pos >= 0:
                insert_pos = nl_pos + 1
            content = content[:insert_pos] + init_code + content[insert_pos:]
            print("  [OK] Inserted SemanticSLAM init after mpTracker creation")
        else:
            print("  [WARN] Could not find end of mpTracker initialization")
    else:
        print("  [WARN] mpTracker initialization not found")

    # 6c. Add shutdown code at the beginning of System::Shutdown()
    shutdown_code = (
        '    // SEMANTIC_SLAM_PATCH_SYSTEM_CC: Shutdown SemanticSLAM\n'
        '    if(mpSemanticSLAM) {\n'
        '        mpSemanticSLAM->stop();\n'
        '        delete mpSemanticSLAM;\n'
        '        mpSemanticSLAM = nullptr;\n'
        '    }\n'
        '\n'
    )

    # Find "void System::Shutdown()" and insert after the opening brace
    shutdown_pattern = re.compile(r'(void\s+System::Shutdown\s*\(\s*\)\s*\{)')
    m = shutdown_pattern.search(content)
    if m:
        insert_pos = m.end()
        content = content[:insert_pos] + '\n' + shutdown_code + content[insert_pos:]
        print("  [OK] Inserted SemanticSLAM shutdown in System::Shutdown()")
    else:
        print("  [WARN] System::Shutdown() not found")

    write_file(filepath, content)


# ============================================================================
# Patch 7: Optimizer.cc — Use semantic weights in LocalBA
#
# In LocalBundleAdjustment, after setting the robust kernel on each edge,
# check if the MapPoint has a semantic weight < 1.0 and increase the
# robust kernel threshold accordingly. This reduces the influence of
# potentially dynamic map points on the optimization.
# ============================================================================
def patch_optimizer_cc():
    filepath = os.path.join(ORB_SLAM3_ROOT, "src", "Optimizer.cc")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_OPTIMIZER"):
        print(f"  [SKIP] {filepath} already patched")
        return

    backup_file(filepath)

    # 7a. Add include for MapPoint.h (needed for HasSemanticClass/GetSemanticWeight)
    # Note: MapPoint.h is likely already included via Optimizer.h, but be explicit
    content = content.replace(
        '#include "Optimizer.h"',
        '#include "Optimizer.h"\n'
        '\n'
        '// SEMANTIC_SLAM_PATCH_OPTIMIZER: Semantic weights in optimization\n'
        '#include "MapPoint.h"\n'
    )

    # 7b. In BundleAdjustment, find the pattern where robust kernel is set
    # for monocular edges and inject semantic weight adjustment.
    #
    # ORB-SLAM3 BundleAdjustment uses:
    #   g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    #   e->setRobustKernel(rk);
    #   rk->setDelta(thHuber2D);     // monocular
    # or:
    #   rk->setDelta(thHuber3D);     // stereo
    #
    # We inject AFTER the setDelta line to override the threshold for dynamic points.

    weight_inject_mono = (
        '                    // SEMANTIC_SLAM_PATCH_OPTIMIZER: Apply semantic weight (mono)\n'
        '                    if(pMP->HasSemanticClass()) {\n'
        '                        double sw = pMP->GetSemanticWeight();\n'
        '                        if(sw < 1.0) {\n'
        '                            rk->setDelta(static_cast<float>(thHuber2D / sw));\n'
        '                        }\n'
        '                    }\n'
    )

    weight_inject_stereo = (
        '                    // SEMANTIC_SLAM_PATCH_OPTIMIZER: Apply semantic weight (stereo)\n'
        '                    if(pMP->HasSemanticClass()) {\n'
        '                        double sw = pMP->GetSemanticWeight();\n'
        '                        if(sw < 1.0) {\n'
        '                            rk->setDelta(static_cast<float>(thHuber3D / sw));\n'
        '                        }\n'
        '                    }\n'
    )

    # Find BundleAdjustment function (not LocalBundleAdjustment which doesn't exist)
    ba_start = content.find("void Optimizer::BundleAdjustment(")
    if ba_start < 0:
        # Try alternate signature
        ba_start = content.find("void Optimizer::BundleAdjustment (")
    if ba_start < 0:
        print("  [WARN] BundleAdjustment not found in Optimizer.cc")
        write_file(filepath, content)
        return

    # Find the end of BundleAdjustment (next function or end of file)
    next_func = re.search(r'\nvoid\s+Optimizer::', content[ba_start + 1:])
    if next_func:
        ba_end = ba_start + 1 + next_func.start()
    else:
        ba_end = len(content)

    ba_body = content[ba_start:ba_end]

    # Patch monocular edges: find "rk->setDelta(thHuber2D);"
    mono_pattern = r'(rk->setDelta\(thHuber2D\);)'
    mono_match = re.search(mono_pattern, ba_body)
    if mono_match:
        ba_body = ba_body[:mono_match.end()] + '\n' + weight_inject_mono + ba_body[mono_match.end():]
        print("  [OK] Patched monocular robust kernel in BundleAdjustment")
    else:
        print("  [WARN] thHuber2D pattern not found in BundleAdjustment")

    # Patch stereo edges (re-search since positions may have shifted)
    stereo_pattern = r'(rk->setDelta\(thHuber3D\);)'
    stereo_match = re.search(stereo_pattern, ba_body)
    if stereo_match:
        ba_body = ba_body[:stereo_match.end()] + '\n' + weight_inject_stereo + ba_body[stereo_match.end():]
        print("  [OK] Patched stereo robust kernel in BundleAdjustment")
    else:
        print("  [WARN] thHuber3D pattern not found in BundleAdjustment")

    content = content[:ba_start] + ba_body + content[ba_end:]

    write_file(filepath, content)


# ============================================================================
# Patch 7b: Optimizer.cc — Fix PoseOptimization g2o crash (BUG FIX #2)
#
# The patched ORB-SLAM3 crashes with SIGSEGV in PoseOptimization:
#   Eigen::LDLT::compute() -> g2o::LinearSolverDense::solve()
#
# Root cause: LinearSolverDense uses LDLT decomposition which requires
# positive-definite matrices. When the Hessian becomes singular (e.g.,
# due to degenerate features), LDLT crashes with SIGSEGV instead of
# returning a numerical error.
#
# Fix: Replace LinearSolverDense with LinearSolverEigen in PoseOptimization.
# LinearSolverEigen uses ColPivHouseholderQR for dense matrices, which is
# robust to singular matrices and will not crash.
# ============================================================================
def patch_optimizer_pose_optimization():
    filepath = os.path.join(ORB_SLAM3_ROOT, "src", "Optimizer.cc")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_POSEOPT"):
        print(f"  [SKIP] {filepath} PoseOptimization already patched")
        return

    backup_file(filepath)

    # Replace LinearSolverDense with LinearSolverEigen in PoseOptimization
    # Pattern in ORB-SLAM3:
    #   typedef g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType> LinearSolverType;
    #   linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    #
    # BUG FIX: This string appears TWICE in PoseOptimization (typedef + new expr).
    # Using replace_all=True ensures both occurrences are updated.
    # The SEMANTIC_SLAM_PATCH_POSEOPT marker is injected on the PREVIOUS line
    # to avoid breaking C++ syntax (// inline comment would eat the rest of line).
    old_solver = "g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>"
    new_solver = "g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>"

    if old_solver in content:
        # Inject patch marker on the line BEFORE the replacement, preserving C++ syntax
        content = content.replace(
            old_solver,
            new_solver,
        )
        # Now inject the marker comment on a separate line before the first occurrence
        marker = "// SEMANTIC_SLAM_PATCH_POSEOPT: replaced LinearSolverDense->Eigen (fix g2o SIGSEGV)\n"
        # Find the first occurrence of new_solver and insert marker above it
        first_pos = content.find("g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>")
        if first_pos >= 0:
            # Find start of the line containing this occurrence
            line_start = content.rfind('\n', 0, first_pos)
            if line_start >= 0:
                content = content[:line_start + 1] + marker + content[line_start + 1:]
            else:
                content = marker + content
        print("  [OK] Replaced LinearSolverDense with LinearSolverEigen in PoseOptimization")
    else:
        # Try alternative pattern (older ORB-SLAM3 versions)
        old_solver2 = "new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>()"
        if old_solver2 in content:
            content = content.replace(
                old_solver2,
                "new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>()"
            )
            # Inject marker comment on a separate line
            first_pos = content.find("g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>")
            if first_pos >= 0:
                line_start = content.rfind('\n', 0, first_pos)
                marker = "// SEMANTIC_SLAM_PATCH_POSEOPT: replaced LinearSolverDense->Eigen (fix g2o SIGSEGV)\n"
                if line_start >= 0:
                    content = content[:line_start + 1] + marker + content[line_start + 1:]
                else:
                    content = marker + content
            print("  [OK] Replaced LinearSolverDense (alt pattern) with LinearSolverEigen")
        else:
            print("  [WARN] LinearSolverDense pattern not found in Optimizer.cc")

    write_file(filepath, content)


# ============================================================================
# Patch 7c: Optimizer.cc — Fix OptimizeSim3 g2o crash (BUG FIX #4)
#
# OptimizeSim3 also uses LinearSolverDense with BlockSolverX,
# causing the same Eigen::LDLT::compute SIGSEGV as PoseOptimization
# when the Hessian matrix becomes singular during loop closing.
#
# Backtrace from deploylog (lines 22494-22511):
#   Eigen::LDLT::compute() -> g2o::LinearSolverDense::solve()
#     -> OptimizationAlgorithmLevenberg::solve() -> SparseOptimizer::optimize()
#     -> Optimizer::OptimizeSim3() -> LoopClosing::DetectCommonRegionsFromBoW()
#     -> LoopClosing::NewDetectCommonRegions() -> LoopClosing::Run()
#
# Fix: Replace LinearSolverDense with LinearSolverEigen in OptimizeSim3.
# BlockSolverX is used only in OptimizeSim3 (Global/Local BA use BlockSolver_6_3),
# so replacing all BlockSolverX occurrences is safe.
# ============================================================================
def patch_optimizer_sim3():
    filepath = os.path.join(ORB_SLAM3_ROOT, "src", "Optimizer.cc")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_SIM3"):
        print(f"  [SKIP] {filepath} OptimizeSim3 already patched")
        return

    backup_file(filepath)

    # OptimizeSim3 uses BlockSolverX (variable-size blocks for Sim3 = 7 dof)
    # Pattern in ORB-SLAM3:
    #   g2o::BlockSolverX::LinearSolverType * linearSolver;
    #   linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    old_solver = "g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>"
    new_solver = "g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>"

    if old_solver in content:
        content = content.replace(old_solver, new_solver)
        # Inject marker comment on a separate line before the first occurrence
        first_pos = content.find(new_solver)
        if first_pos >= 0:
            line_start = content.rfind('\n', 0, first_pos)
            marker = "// SEMANTIC_SLAM_PATCH_SIM3: replaced LinearSolverDense->Eigen (fix OptimizeSim3 SIGSEGV)\n"
            if line_start >= 0:
                content = content[:line_start + 1] + marker + content[line_start + 1:]
            else:
                content = marker + content
        print("  [OK] Replaced LinearSolverDense with LinearSolverEigen in OptimizeSim3")
    else:
        # Try alternative pattern (older ORB-SLAM3 versions)
        old_solver2 = "new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>()"
        if old_solver2 in content:
            content = content.replace(
                old_solver2,
                "new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>()"
            )
            first_pos = content.find("g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>")
            if first_pos >= 0:
                line_start = content.rfind('\n', 0, first_pos)
                marker = "// SEMANTIC_SLAM_PATCH_SIM3: replaced LinearSolverDense->Eigen (fix OptimizeSim3 SIGSEGV)\n"
                if line_start >= 0:
                    content = content[:line_start + 1] + marker + content[line_start + 1:]
                else:
                    content = marker + content
            print("  [OK] Replaced LinearSolverDense (alt pattern) with LinearSolverEigen in OptimizeSim3")
        else:
            print("  [WARN] BlockSolverX/LinearSolverDense pattern not found in Optimizer.cc")

    write_file(filepath, content)


# ============================================================================
# Patch 7d: MLPnPsolver.cpp — Add minimum point guard (BUG FIX #5)
#
# Backtrace from deploylog (lines 23914-23930, 24664-24680, 24773-24789):
#   cfree -> MLPnPsolver::computePose -> MLPnPsolver::iterate
#     -> Tracking::Relocalization -> Tracking::Track -> System::TrackStereo
#
# Root cause: When dynamic feature filtering reduces the number of valid
# matches, MLPnPsolver::computePose receives degenerate point sets that
# cause use-after-free in Eigen aligned_allocator within the algorithm.
#
# Fix: Add a minimum point count guard at the entry of computePose().
# MLPnP requires at least 6 points for a valid pose estimate.
#
# IMPORTANT: ORB-SLAM3's MLPnPsolver::computePose has UNNAMED parameters
# and returns void. We must first name the parameters, then add the guard.
# Actual signature (from compiler error):
#   void MLPnPsolver::computePose(const bearingVectors_t&, const points_t&,
#       const cov3_mats_t&, const std::vector<int>&, transformation_t&)
# ============================================================================
def patch_mlpnpsolver():
    filepath = os.path.join(ORB_SLAM3_ROOT, "src", "MLPnPsolver.cpp")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_MLPNP"):
        print(f"  [SKIP] {filepath} already patched")
        return

    backup_file(filepath)

    import re as mlpnp_re

    # Step 1: Add parameter names to the unnamed function signature.
    # The signature uses unnamed parameters; we add names so we can reference
    # the points_t parameter (p) in the guard.
    #
    # We use a flexible regex to match the signature regardless of whitespace:
    old_sig_pat = (
        r'void\s+MLPnPsolver::computePose\s*\(\s*'
        r'const\s+bearingVectors_t\s*&\s*,'
        r'\s*const\s+points_t\s*&\s*,'
        r'\s*const\s+cov3_mats_t\s*&\s*,'
        r'\s*const\s+std::vector<int>\s*&\s*,'
        r'\s*transformation_t\s*&\s*\)'
    )

    new_sig = (
        'void MLPnPsolver::computePose('
        'const bearingVectors_t& f, '
        'const points_t& p, '
        'const cov3_mats_t& covMats, '
        'const std::vector<int>& indices, '
        'transformation_t& result)'
    )

    if mlpnp_re.search(old_sig_pat, content):
        content = mlpnp_re.sub(old_sig_pat, new_sig, content, count=1)
        print("  [OK] Named MLPnPsolver::computePose parameters")
    else:
        # Fallback: try to match with different spacing/formatting
        # The function might already have named parameters (older ORB-SLAM3)
        old_sig_pat2 = (
            r'void\s+MLPnPsolver::computePose\s*\(\s*'
            r'const\s+bearingVectors_t\s*&\s*\w+\s*,'
            r'\s*const\s+points_t\s*&\s*\w+\s*,'
        )
        if mlpnp_re.search(old_sig_pat2, content):
            print("  [OK] MLPnPsolver::computePose already has named params")
        else:
            print("  [WARN] MLPnPsolver::computePose signature not found (unexpected format)")

    # Step 2: Insert the guard after the opening brace of computePose()
    func_pattern = "MLPnPsolver::computePose("
    func_pos = content.find(func_pattern)
    if func_pos >= 0:
        brace_pos = content.find('{', func_pos)
        if brace_pos >= 0:
            nl = content.find('\n', brace_pos)
            if nl >= 0:
                guard = (
                    '\n'
                    '    // SEMANTIC_SLAM_PATCH_MLPNP: Guards for degenerate inputs\n'
                    '    // MLPnP requires at least 6 points. Fewer points indicate\n'
                    '    // dynamic filtering has removed too many features, causing\n'
                    '    // use-after-free in Eigen aligned_allocator (SIGSEGV).\n'
                    '    if(f.size() < 6 || p.size() < 6) {\n'
                    '        return;\n'
                    '    }\n'
                    '    // SEMANTIC_SLAM_PATCH_MLPNP: Consistency check — input vectors\n'
                    '    // must have matching sizes. Mismatch indicates corrupted data\n'
                    '    // from dynamic feature filtering, which causes heap corruption\n'
                    '    // in Eigen aligned_allocator (cfree SIGSEGV at +0x326c).\n'
                    '    if(f.size() != p.size() || f.size() != covMats.size()) {\n'
                    '        return;\n'
                    '    }\n'
                    '    // SEMANTIC_SLAM_PATCH_MLPNP: Validity check — all indices must\n'
                    '    // be in range. Invalid indices cause out-of-bounds access in\n'
                    '    // Eigen matrices (cfree SIGSEGV).\n'
                    '    for(int idx : indices) {\n'
                    '        if(idx < 0 || idx >= (int)f.size()) {\n'
                    '            return;\n'
                    '        }\n'
                    '    }\n'
                    '    // SEMANTIC_SLAM_PATCH_MLPNP: NaN check — NaN values in bearing\n'
                    '    // vectors or 3D points cause degenerate computations in MLPnP\n'
                    '    // leading to heap corruption in Eigen (cfree SIGSEGV).\n'
                    '    for(size_t i = 0; i < f.size(); ++i) {\n'
                    '        if(!std::isfinite(f[i](0)) || !std::isfinite(f[i](1)) || !std::isfinite(f[i](2)))\n'
                    '            return;\n'
                    '        if(!std::isfinite(p[i](0)) || !std::isfinite(p[i](1)) || !std::isfinite(p[i](2)))\n'
                    '            return;\n'
                    '    }\n'
                    '    // SEMANTIC_SLAM_PATCH_MLPNP: Covariance NaN check — degenerate\n'
                    '    // covariance matrices (from near-zero-depth points) cause\n'
                    '    // Eigen LDLT factorization failures in null-space computation,\n'
                    '    // leading to heap corruption (cfree SIGSEGV at +0x326c).\n'
                    '    for(size_t i = 0; i < covMats.size(); ++i) {\n'
                    '        for(int r = 0; r < 3; ++r) {\n'
                    '            for(int c = 0; c < 3; ++c) {\n'
                    '                if(!std::isfinite(covMats[i](r, c)))\n'
                    '                    return;\n'
                    '            }\n'
                    '        }\n'
                    '    }\n'
                    '\n'
                )
                content = content[:nl + 1] + guard + content[nl + 1:]
                print("  [OK] Added comprehensive guards in MLPnPsolver::computePose")
            else:
                print("  [WARN] No newline after MLPnPsolver::computePose opening brace")
        else:
            print("  [WARN] No opening brace found in MLPnPsolver::computePose")
    else:
        print("  [WARN] MLPnPsolver::computePose not found")

    write_file(filepath, content)


# ============================================================================
# Patch 8: ORBmatcher.cc — Skip dynamic features in matching
#
# In SearchByProjection and SearchByBoW, skip keypoints that are marked
# as dynamic. This prevents dynamic features from being matched and
# creating incorrect map point associations.
# ============================================================================
def patch_orbmatcher_cc():
    filepath = os.path.join(ORB_SLAM3_ROOT, "src", "ORBmatcher.cc")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_MATCHER"):
        print(f"  [SKIP] {filepath} already patched")
        return

    backup_file(filepath)

    # 8a. Add include
    content = content.replace(
        '#include "ORBmatcher.h"',
        '#include "ORBmatcher.h"\n'
        '\n'
        '// SEMANTIC_SLAM_PATCH_MATCHER: Skip dynamic features\n'
        '#include "Frame.h"\n'
    )

    # 8b. Add dynamic feature skip in SearchByProjection
    # ORB-SLAM3 SearchByProjection has inner loop:
    #   for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
    #   {
    #       const size_t idx = *vit;
    #       if(F.mvpMapPoints[idx])
    #           if(F.mvpMapPoints[idx]->Observations()>0)
    #               continue;
    # We add dynamic check BEFORE the mvpMapPoints check using idx.
    dynamic_check_idx = (
        '                        // SEMANTIC_SLAM_PATCH_MATCHER: Skip dynamic features\n'
        '                        if(F.IsDynamicFeature(idx))\n'
        '                            continue;\n'
    )
    # Match SearchByProjection — use [\s\S]*? instead of [^)]* to handle nested templates
    pattern_sbp = r'(int\s+ORBmatcher::SearchByProjection\s*\([\s\S]*?Frame\s+&F[\s\S]*?\))'
    m_sbp = re.search(pattern_sbp, content)
    if m_sbp:
        # Find the first "if(F.mvpMapPoints[idx])" after the function start
        search_start = m_sbp.end()
        mvp_match = re.search(r'if\s*\(\s*F\.mvpMapPoints\[idx\]\s*\)', content[search_start:])
        if mvp_match:
            insert_pos = search_start + mvp_match.start()
            # Find the start of the line containing this if
            line_start = content.rfind('\n', 0, insert_pos) + 1
            content = content[:line_start] + dynamic_check_idx + content[line_start:]
            print("  [OK] Patched SearchByProjection with dynamic feature skip")
        else:
            print("  [WARN] F.mvpMapPoints[idx] pattern not found in SearchByProjection")
    else:
        print("  [WARN] SearchByProjection(Frame&) not found")

    # 8c. Add dynamic feature skip in SearchByBoW(KeyFrame*, Frame&, ...)
    # ORB-SLAM3 SearchByBoW uses Frame &F, with variable realIdxF for indices.
    # The pattern is: if(vpMapPointMatches[realIdxF])
    # We add dynamic check before the first such check.
    dynamic_check_realf = (
        '                            // SEMANTIC_SLAM_PATCH_MATCHER: Skip dynamic features\n'
        '                            if(F.IsDynamicFeature(realIdxF))\n'
        '                                continue;\n'
    )
    # Use [\s\S]*? to handle nested templates in function signature
    pattern_sbw = r'(int\s+ORBmatcher::SearchByBoW\s*\(\s*KeyFrame\*[\s\S]*?Frame\s+&F[\s\S]*?\))'
    m_sbw = re.search(pattern_sbw, content)
    if m_sbw:
        search_start = m_sbw.end()
        # Find the first "if(vpMapPointMatches[realIdxF])" after function start
        mvp_match = re.search(r'if\s*\(\s*vpMapPointMatches\[realIdxF\]\s*\)', content[search_start:])
        if mvp_match:
            insert_pos = search_start + mvp_match.start()
            line_start = content.rfind('\n', 0, insert_pos) + 1
            content = content[:line_start] + dynamic_check_realf + content[line_start:]
            print("  [OK] Patched SearchByBoW with dynamic feature skip")
        else:
            print("  [WARN] vpMapPointMatches[realIdxF] pattern not found in SearchByBoW")
    else:
        print("  [WARN] SearchByBoW(KeyFrame*, Frame&) not found")

    write_file(filepath, content)


# ============================================================================
# Patch 9: CMakeLists.txt — Link SemanticSLAM library
# ============================================================================
def patch_cmakelists():
    filepath = os.path.join(ORB_SLAM3_ROOT, "CMakeLists.txt")
    content = read_file(filepath)

    if is_patched(content, "SEMANTIC_SLAM_PATCH_CMAKE"):
        print(f"  [SKIP] {filepath} already patched")
        return

    backup_file(filepath)

    # Add SemanticSLAM include directory and library path
    # NOTE: SEMANTIC_SLAM_ROOT is the src directory (passed as 2nd arg)
    # CRITICAL: Insert set(SEMANTIC_SLAM_DIR ...) after project() to ensure it's in
    # the TOP-LEVEL scope, NOT inside any if/else conditional block.
    sem_include = (
        f"\n# SEMANTIC_SLAM_PATCH_CMAKE: SemanticSLAM include & library\n"
        f"set(SEMANTIC_SLAM_DIR \"{SEMANTIC_SLAM_ROOT}\")\n"
        f"include_directories(${{SEMANTIC_SLAM_DIR}}/include)\n"
    )

    # Insert after project() command — always top-level, never inside if/else
    proj_match = re.search(r'project\s*\([^)]*\)', content)
    if proj_match:
        insert_pos = proj_match.end()
        nl = content.find('\n', insert_pos)
        if nl >= 0:
            insert_pos = nl + 1
        content = content[:insert_pos] + sem_include + content[insert_pos:]
        print("  [OK] Inserted SEMANTIC_SLAM_DIR after project()")
    elif "include_directories(" in content:
        # Fallback: insert after the last include_directories call
        last_inc = content.rfind("include_directories(")
        close_paren = content.find(")", last_inc)
        if close_paren >= 0:
            insert_pos = close_paren + 1
            nl = content.find('\n', insert_pos)
            if nl >= 0:
                insert_pos = nl + 1
            content = content[:insert_pos] + sem_include + content[insert_pos:]
            print("  [OK] Inserted SEMANTIC_SLAM_DIR after include_directories (fallback)")
    else:
        # Last resort: insert at the beginning
        content = sem_include + content
        print("  [WARN] Inserted SEMANTIC_SLAM_DIR at file beginning (last resort)")

    # Add library link to target_link_libraries — use FULL PATH to avoid link_directories issues
    sem_link = f" ${{SEMANTIC_SLAM_DIR}}/build/libSemanticSLAM.so"
    if "target_link_libraries(" in content:
        # Find the main target_link_libraries for ORB_SLAM3
        # Look for the one that links the main library
        tl_match = re.search(r'target_link_libraries\s*\(\s*\$\{PROJECT_NAME\}', content)
        if not tl_match:
            tl_match = re.search(r'target_link_libraries\s*\(\s*ORB_SLAM3', content)
        if tl_match:
            # Find the closing paren
            pos = tl_match.start()
            close = content.find(')', pos)
            if close >= 0:
                content = content[:close] + sem_link + content[close:]
                print("  [OK] Added SemanticSLAM to target_link_libraries")

    write_file(filepath, content)


# ============================================================================
# Main
# ============================================================================
def main():
    global ORB_SLAM3_ROOT, SEMANTIC_SLAM_ROOT

    if len(sys.argv) < 2:
        print("Usage: python3 patch_orbslam3.py <ORB_SLAM3_ROOT> [SEMANTIC_SLAM_ROOT]")
        print("Example: python3 patch_orbslam3.py /root/ORB_SLAM3 ~/autodl-tmp/Script/semantic-slam-yolov8/src")
        sys.exit(1)

    ORB_SLAM3_ROOT = sys.argv[1]
    SEMANTIC_SLAM_ROOT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/autodl-tmp/Script/semantic-slam-yolov8/src")

    if not os.path.isdir(ORB_SLAM3_ROOT):
        print(f"ERROR: ORB_SLAM3_ROOT not found: {ORB_SLAM3_ROOT}")
        sys.exit(1)

    print("=" * 60)
    print("Patching ORB-SLAM3 for SemanticSLAM Integration")
    print("=" * 60)
    print(f"  ORB-SLAM3:     {ORB_SLAM3_ROOT}")
    print(f"  SemanticSLAM:  {SEMANTIC_SLAM_ROOT}")
    print()

    # Copy SemanticSLAM headers to ORB-SLAM3 include directory
    print("==> Copying SemanticSLAM headers to ORB-SLAM3...")
    target_include = os.path.join(ORB_SLAM3_ROOT, "include")
    for hfile in ["SemanticSLAM.h", "YoloDetector.h", "DynamicFeatureFilter.h",
                   "SemanticWeights.h"]:
        src = os.path.join(SEMANTIC_SLAM_ROOT, "include", hfile)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(target_include, hfile))
            print(f"  [COPY] {hfile}")
        else:
            print(f"  [WARN] Header not found: {hfile}")

    # Apply patches
    print("\n==> Patching source files...")
    patch_frame_h()
    patch_frame_cc()
    patch_mappoint_h()
    patch_tracking_h()
    patch_system_h()
    patch_tracking_cc()
    patch_system_cc()
    patch_optimizer_cc()
    patch_optimizer_pose_optimization()
    patch_optimizer_sim3()
    patch_mlpnpsolver()
    patch_orbmatcher_cc()
    patch_cmakelists()

    print("\n" + "=" * 60)
    print("All patches applied!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. cd /root/ORB_SLAM3/build")
    print("  2. cmake .. -DCMAKE_BUILD_TYPE=Release")
    print("  3. make -j$(nproc)")
    print()
    print("To revert patches:")
    print("  for f in $(find . -name '*.orig'); do mv $f ${f%.orig}; done")


if __name__ == "__main__":
    main()
