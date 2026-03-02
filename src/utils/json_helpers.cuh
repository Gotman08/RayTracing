/**
 * JSON Parsing Helpers for Ray Tracer
 * Provides clean conversion from JSON arrays to Vec3/Point3/Color types
 */

#ifndef RAYTRACER_UTILS_JSON_HELPERS_CUH
#define RAYTRACER_UTILS_JSON_HELPERS_CUH

#include "json.hpp"
#include "raytracer/core/vec3.cuh"

using json = nlohmann::json;

namespace rt {

// Extract Point3 from JSON with default value
inline Point3 json_to_point3(const json& j, const std::string& key, const Point3& default_val) {
    if (j.contains(key)) {
        auto v = j[key].get<std::vector<float>>();
        if (v.size() >= 3) {
            return Point3(v[0], v[1], v[2]);
        }
    }
    return default_val;
}

// Extract Vec3 from JSON with default value
inline Vec3 json_to_vec3(const json& j, const std::string& key, const Vec3& default_val) {
    if (j.contains(key)) {
        auto v = j[key].get<std::vector<float>>();
        if (v.size() >= 3) {
            return Vec3(v[0], v[1], v[2]);
        }
    }
    return default_val;
}

// Extract Color from JSON with default value
inline Color json_to_color(const json& j, const std::string& key, const Color& default_val) {
    if (j.contains(key)) {
        auto v = j[key].get<std::vector<float>>();
        if (v.size() >= 3) {
            return Color(v[0], v[1], v[2]);
        }
    }
    return default_val;
}

// Overload for direct JSON array (without key lookup)
inline Point3 json_array_to_point3(const json& arr, const Point3& default_val = Point3(0, 0, 0)) {
    if (arr.is_array() && arr.size() >= 3) {
        return Point3(arr[0], arr[1], arr[2]);
    }
    return default_val;
}

inline Vec3 json_array_to_vec3(const json& arr, const Vec3& default_val = Vec3(0, 0, 0)) {
    if (arr.is_array() && arr.size() >= 3) {
        return Vec3(arr[0], arr[1], arr[2]);
    }
    return default_val;
}

inline Color json_array_to_color(const json& arr, const Color& default_val = Color(0, 0, 0)) {
    if (arr.is_array() && arr.size() >= 3) {
        return Color(arr[0], arr[1], arr[2]);
    }
    return default_val;
}

} // namespace rt

#endif // RAYTRACER_UTILS_JSON_HELPERS_CUH
